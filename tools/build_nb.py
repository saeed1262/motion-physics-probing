import nbformat as nbf

cells = []  # (kind, tag, source); tag "setup" cells are skipped by the local test


def md(s):
    cells.append(("md", None, s.strip("\n")))


def code(s, tag="main"):
    cells.append(("code", tag, s.strip("\n")))


md(r"""
# Do motion models know about balance? (feasibility test)

This notebook looks inside **MDM** (the Motion Diffusion Model) while it generates human motion from text, and asks one question:

> Can we read **physical quantities of the final motion** (balance, floating, flight) from the model's hidden activations, and how early in the denoising process?

**What it does**
1. Installs MDM and downloads the 50-step pretrained model (no AMASS registration needed).
2. Generates 256 motions from 64 prompts chosen to stress balance and jumping, recording every transformer layer at 6 denoising steps.
3. Computes physics labels per frame: gravity during flight, dynamic balance (extrapolated centre of mass vs. support polygon), static balance, lowest-foot height (floating), and in-flight.
4. Trains linear probes from activations to those labels, against three baselines.
5. Plots where (layer) and when (denoising step) the physics becomes readable.
6. Tests **early failure prediction**: can early-step activations predict which samples will float or lose balance, better than the prompt alone?

**How to run (works from a phone browser)**
1. `Runtime` → `Change runtime type` → **T4 GPU** → Save.
2. `Runtime` → **Run all**.
3. Expect roughly 10–15 minutes, mostly downloads. Results appear at the bottom.
""")

md("## 1. Setup")

code(r"""
import os, subprocess, glob

def sh(cmd):
    # Run a shell command; show output only if it fails.
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"Command failed: {cmd}")
    return r.stdout

print(sh("nvidia-smi -L || echo 'No GPU found: use Runtime > Change runtime type > T4 GPU'"))

os.chdir("/content")
if not os.path.exists("/content/mdm"):
    sh("git clone -q https://github.com/GuyTevet/motion-diffusion-model.git mdm")
if not os.path.exists("/content/hml"):
    # Only needed for Mean.npy / Std.npy (normalisation stats). No motion data is used.
    sh("git clone -q --depth 1 https://github.com/EricGuo5513/HumanML3D.git hml")
sh("pip -q install git+https://github.com/openai/CLIP.git gdown shapely")
print("Setup done.")
""", tag="setup")

code(r"""
# Pretrained MDM: "humanml-encoder-512-50steps" from the official MDM README.
CKPT_URL = "https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing"
# If Google Drive refuses the download (quota), put the zip in your own Drive,
# set LOCAL_ZIP below to its path, and rerun this cell.
LOCAL_ZIP = ""   # e.g. "/content/drive/MyDrive/humanml_enc_512_50steps.zip"

os.makedirs("/content/mdm/save", exist_ok=True)
if not glob.glob("/content/mdm/save/**/args.json", recursive=True):
    if LOCAL_ZIP:
        if LOCAL_ZIP.startswith("/content/drive"):
            from google.colab import drive
            drive.mount("/content/drive")
        sh(f"cp '{LOCAL_ZIP}' /content/mdm/save/ckpt.zip")
    else:
        try:
            sh(f"gdown --fuzzy '{CKPT_URL}' -O /content/mdm/save/ckpt.zip")
        except RuntimeError:
            raise RuntimeError(
                "Google Drive blocked the download (common quota limit). Open CKPT_URL in a browser, "
                "choose 'Add shortcut to Drive' or download it into your Drive, then set LOCAL_ZIP and rerun.")
    sh("cd /content/mdm/save && unzip -q -o ckpt.zip && rm ckpt.zip")
print(glob.glob("/content/mdm/save/**/*", recursive=True)[:10])
""", tag="setup")

md("""
## 2. Load MDM

MDM normally builds an SMPL body layer and loads GloVe vectors, but text-to-motion on HumanML3D outputs joint positions directly and never uses them. A small stub skips those downloads.
""")

code(r"""
import sys, types, json
import numpy as np
import torch, torch.nn as nn

MDM_DIR = os.environ.get("MDM_DIR", "/content/mdm")
HML_DIR = os.environ.get("HML_DIR", "/content/hml/HumanML3D")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.chdir(MDM_DIR)
sys.path.insert(0, MDM_DIR)

# Stub out the SMPL-based rotation layer (unused for HumanML3D joint-position output).
_stub = types.ModuleType("model.rotation2xyz")
class _Rot2xyzStub:
    def __init__(self, device="cpu", dataset=None):
        self.smpl_model = nn.Identity()
    def __call__(self, x, *args, **kwargs):
        return x
_stub.Rotation2xyz = _Rot2xyzStub
sys.modules["model.rotation2xyz"] = _stub

# Compatibility: MDM imports scipy.ndimage.filters, which future SciPy versions may drop.
try:
    import scipy.ndimage.filters  # noqa: F401
except ImportError:
    import scipy.ndimage
    sys.modules["scipy.ndimage.filters"] = scipy.ndimage

from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.humanml.scripts.motion_process import recover_from_ric

args_path = glob.glob(os.path.join(MDM_DIR, "save/**/args.json"), recursive=True)[0]
ckpt_path = sorted(glob.glob(os.path.join(os.path.dirname(args_path), "model*.pt")))[-1]
defaults = dict(dataset="humanml", unconstrained=False, arch="trans_enc", text_encoder_type="clip",
                latent_dim=512, layers=8, cond_mask_prob=0.1, emb_trans_dec=False, pos_embed_max_len=5000,
                mask_frames=False, diffusion_steps=1000, noise_schedule="cosine", sigma_small=True,
                lambda_vel=0.0, lambda_rcxyz=0.0, lambda_fc=0.0, lambda_target_loc=0.0, use_ema=False)
with open(args_path) as f:
    saved_args = json.load(f)
args = types.SimpleNamespace(**{**defaults, **saved_args})

class _NoData:
    dataset = types.SimpleNamespace()   # get_model_args only checks it for num_actions

model, diffusion = create_model_and_diffusion(args, _NoData())
if os.environ.get("SKIP_CKPT") != "1":
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # official checkpoint
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model_avg"] if (args.use_ema and "model_avg" in sd) else sd["model"]
    load_model_wo_clip(model, sd)
model.to(DEVICE)   # note: MDM.to() returns None, so no chaining
model.eval()

MEAN = torch.tensor(np.load(os.path.join(HML_DIR, "Mean.npy"))).float()
STD = torch.tensor(np.load(os.path.join(HML_DIR, "Std.npy"))).float()
LAYERS = list(model.seqTransEncoder.layers)
print(f"Loaded {os.path.basename(ckpt_path)} | arch={args.arch} | layers={len(LAYERS)} | "
      f"diffusion steps={diffusion.num_timesteps} | device={DEVICE}")
""")

md("""
## 3. Generate motions and record activations

Prompts are grouped so the set contains plenty of jumps (flight) and balance-challenging poses. Each prompt is sampled `N_REPS` times with different noise.
""")

code(r"""
PROMPTS = [
    # everyday locomotion
    "a person walks forward.", "a person walks forward slowly and carefully.", "a person jogs forward.",
    "a person runs forward quickly.", "a person walks in a circle.", "a person walks backward.",
    "a person sidesteps to the left.", "a person walks forward and turns around.",
    # flight: jumps, hops, leaps
    "a person jumps up in place.", "a person jumps forward.", "a person does a long jump forward.",
    "a person hops on their left leg.", "a person hops on their right leg.", "a person jumps over an obstacle.",
    "a person skips forward.", "a person does jumping jacks.", "a person jumps up and spins in the air.",
    "a person leaps to the side.", "a person runs and jumps.", "a person jumps backward.",
    "a person jumps with both feet three times.", "a person jumps and lands on one foot.",
    # balance-challenging
    "a person stands on one leg.", "a person balances on their right foot with arms out.",
    "a person leans forward.", "a person leans to the left.", "a person stands on their toes.",
    "a person walks on a balance beam.", "a person loses balance and catches themselves.",
    "a person stumbles forward.", "a person bends down to touch their toes.", "a person squats down and stands up.",
    "a person does a lunge forward.", "a person kicks with their right leg.", "a person kicks high with their left leg.",
    "a person does a yoga tree pose.", "a person tiptoes forward.", "a person runs and then stops suddenly.",
    # standing with upper-body action
    "a person waves with their right hand.", "a person throws a ball.", "a person picks something up from the ground.",
    "a person reaches up high.", "a person claps their hands.", "a person stretches their arms.",
    "a person punches forward.", "a person boxes.", "a person swings a golf club.", "a person kicks a soccer ball.",
    # misc
    "a person dances.", "a person does a spin.", "a person marches in place.", "a person crouches and walks.",
    "a person walks while carrying something heavy.", "a person dribbles a basketball and shoots.",
    "a person plays tennis.", "a person shuffles side to side.", "a person steps forward and back.",
    "a person turns in place.", "a person stands still.", "a person stretches their legs.",
    "a person shakes their body.", "a person runs in a zigzag.", "a person jogs in place.", "a person sprints.",
]
N_REPS = int(os.environ.get("N_REPS", 4))      # samples per prompt
N_FRAMES = int(os.environ.get("N_FRAMES", 120))  # 6 s at 20 fps
BATCH = 64
GUIDANCE = 2.5                                  # MDM's default classifier-free guidance scale
FRAME_STRIDE = 2                                # keep every 2nd frame of activations (memory)
N_STEPS = diffusion.num_timesteps
RECORD_STEPS = sorted(set(np.linspace(0, N_STEPS - 1, 6).round().astype(int).tolist()))
FR = np.arange(0, N_FRAMES, FRAME_STRIDE)       # frames kept for probing
print(f"{len(PROMPTS)} prompts x {N_REPS} reps = {len(PROMPTS) * N_REPS} motions; "
      f"recording denoising steps {RECORD_STEPS} (0 = pure noise, {N_STEPS - 1} = final)")
""")

code(r"""
class Recorder:
    # Forward hooks on every transformer layer; stores outputs only when switched on.
    def __init__(self, layers):
        self.on, self.buf = False, {}
        for li, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(li))

    def _hook(self, li):
        def hook(module, inputs, out):
            if self.on:
                h = out[1:][::FRAME_STRIDE]                 # drop the text/time token -> [F, bs, d]
                self.buf[li] = h.permute(1, 0, 2).to(torch.float16).cpu()
        return hook


class RecordingCFG(nn.Module):
    # Classifier-free guidance (same maths as MDM's wrapper); records the conditional pass only.
    def __init__(self, model, recorder):
        super().__init__()
        self.model, self.recorder, self.record_next = model, recorder, False

    def forward(self, x, timesteps, y=None):
        y_uncond = dict(y, uncond=True)
        self.recorder.on = self.record_next
        out = self.model(x, timesteps, y)
        self.recorder.on = False
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + y["scale"].view(-1, 1, 1, 1) * (out - out_uncond)


recorder = Recorder(LAYERS)
cfg_model = RecordingCFG(model, recorder).eval()


def to_joints(x):
    # Normalised MDM output [bs, 263, 1, T] -> joints [bs, T, 22, 3] (metres, Y up) and raw features.
    v = x[:, :, 0, :].permute(0, 2, 1).float().cpu() * STD + MEAN
    return recover_from_ric(v, 22).numpy(), v.numpy()


@torch.no_grad()
def sample_batch(prompts, seed):
    bs = len(prompts)
    torch.manual_seed(seed)
    y = {"mask": torch.ones(bs, 1, 1, N_FRAMES, dtype=torch.bool, device=DEVICE),
         "lengths": torch.full((bs,), N_FRAMES, device=DEVICE),
         "text": list(prompts),
         "scale": torch.full((bs,), GUIDANCE, device=DEVICE)}
    y["text_embed"] = model.encode_text(list(prompts))
    x = torch.randn((bs, model.njoints, model.nfeats, N_FRAMES), device=DEVICE)
    acts, xt_feats, x0_joints = {}, {}, {}
    for k, t_int in enumerate(range(N_STEPS - 1, -1, -1)):
        t = torch.full((bs,), t_int, device=DEVICE, dtype=torch.long)
        cfg_model.record_next = k in RECORD_STEPS
        recorder.buf = {}
        out = diffusion.p_sample(cfg_model, x, t, clip_denoised=False, model_kwargs={"y": y})
        if k in RECORD_STEPS:
            for li, h in recorder.buf.items():
                acts[(k, li)] = h.numpy()
            xt_feats[k] = x[:, :, 0, ::FRAME_STRIDE].permute(0, 2, 1).to(torch.float16).cpu().numpy()
            x0_joints[k] = to_joints(out["pred_xstart"])   # (joints, features) of the model's current x0 guess
        x = out["sample"]
    joints, feats = to_joints(x)
    # encode_text returns [1, bs, 512]; drop the leading axis so text embeddings stack per sample.
    return joints, feats, acts, xt_feats, x0_joints, y["text_embed"][0].float().cpu().numpy()
""")

code(r"""
import time
all_prompts = [p for _ in range(N_REPS) for p in PROMPTS]
prompt_id = np.array([i for _ in range(N_REPS) for i in range(len(PROMPTS))])

J_final, F_final, TXT, ACTS, XT, X0J, X0F = [], [], [], {}, {}, {}, {}
t0 = time.time()
for b in range(0, len(all_prompts), BATCH):
    j, f, acts, xt, x0, txt = sample_batch(all_prompts[b:b + BATCH], seed=b)
    J_final.append(j); F_final.append(f); TXT.append(txt)
    for key, v in acts.items(): ACTS.setdefault(key, []).append(v)
    for k in RECORD_STEPS:
        XT.setdefault(k, []).append(xt[k])
        X0J.setdefault(k, []).append(x0[k][0])
        X0F.setdefault(k, []).append(x0[k][1][:, FR].astype(np.float16))
    print(f"batch {b // BATCH + 1}/{-(-len(all_prompts) // BATCH)} done, {time.time() - t0:.0f}s")

J_final = np.concatenate(J_final); F_final = np.concatenate(F_final); TXT = np.concatenate(TXT)
ACTS = {k: np.concatenate(v) for k, v in ACTS.items()}
XT = {k: np.concatenate(v) for k, v in XT.items()}
X0J = {k: np.concatenate(v) for k, v in X0J.items()}
X0F = {k: np.concatenate(v) for k, v in X0F.items()}
print("final joints:", J_final.shape, "| activations per (step, layer):", next(iter(ACTS.values())).shape)
""")

md("""
## 4. Physics labels

All labels come from joint positions (Y up, metres, 20 fps):
- **flight gravity** (cleanest physics test): over each airborne stretch of at least 0.2 s (no foot, knee, wrist or pelvis near the ground), we fit a parabola to the center of mass height. Real flight falls at 9.81 m/s²; a body that hovers or drifts gives a smaller number. This needs no calibration.
- **XCoM margin** (main balance quantity, *not* a failure test): like the balance margin below, but using the *extrapolated* centre of mass (position + velocity / ω₀, Hof 2005). During normal stepping it often lies outside the standing foot, because the next step catches the body, so we probe it as a continuous quantity rather than counting violations.
- **balance margin**: signed distance from the ground-projected centre of mass to the support polygon of the feet in contact (positive = inside). Undefined when no foot touches the ground.
- **lowest foot**: height of the lowest foot joint (floating or penetration signal).
- **in flight**: no foot in contact.

Contact uses HumanML3D's own velocity rule plus a height limit. The centre of mass uses standard segment mass fractions.
""")

code(r"""
from shapely.geometry import MultiPoint, Point

FPS = 20.0
FOOT_JOINTS = [7, 10, 8, 11]                       # L ankle, L toe, R ankle, R toe (HumanML3D order)
FOOT_HEIGHT_MAX = np.array([0.12, 0.06, 0.12, 0.06])
FOOT_DISP2_MAX = 0.002                             # HumanML3D contact rule (squared m per frame)
FOOT_RADIUS = 0.04                                 # padding around contact points (m)
SEGMENTS = [(12, 15, 0.081), (0, 3, 0.142), (3, 9, 0.355), (16, 18, 0.028), (17, 19, 0.028),
            (18, 20, 0.016), (19, 21, 0.016), (20, 20, 0.006), (21, 21, 0.006), (1, 4, 0.100),
            (2, 5, 0.100), (4, 7, 0.0465), (5, 8, 0.0465), (7, 10, 0.0145), (8, 11, 0.0145)]


def center_of_mass(j):
    com = np.zeros((j.shape[0], 3))
    for a, b, w in SEGMENTS:
        com += w * 0.5 * (j[:, a] + j[:, b])
    return com


def foot_contacts(j):
    feet = j[:, FOOT_JOINTS]
    d2 = np.zeros(feet.shape[:2])
    d2[1:] = ((feet[1:] - feet[:-1]) ** 2).sum(-1)
    d2[0] = d2[1]
    return (d2 < FOOT_DISP2_MAX) & (feet[..., 1] < FOOT_HEIGHT_MAX[None])


def balance_margin(com_xz, pts_xz):
    if len(pts_xz) == 0:
        return np.nan
    poly = MultiPoint([tuple(p) for p in pts_xz]).convex_hull.buffer(FOOT_RADIUS)
    p = Point(*com_xz)
    return poly.exterior.distance(p) if poly.contains(p) else -poly.distance(p)


MIN_FLIGHT = 4      # airborne frames needed to fit a ballistic arc (0.2 s at 20 fps)
G_EARTH = 9.81
SUPPORT_JOINTS = [0, 4, 5, 20, 21]  # pelvis, knees, wrists: body parts that can carry weight instead of the feet
SUPPORT_HEIGHT_MIN = 0.15           # any of them this close to the floor means supported, not airborne
FLOAT_G_MAX = 2.0                   # an unsupported stretch falling slower than this (m/s^2) is floating, not a jump


def flight_gravity(com_y, airborne):
    # Fit y(t) = c0 + c1 t + c2 t^2 to the center of mass over each airborne stretch.
    # Real ballistic flight gives g = -2 c2 = 9.81 m/s^2; a body that hovers gives g near 0.
    g = np.full(len(com_y), np.nan)
    t, n = 0, len(com_y)
    while t < n:
        if not airborne[t]:
            t += 1
            continue
        s = t
        while t < n and airborne[t]:
            t += 1
        if t - s >= MIN_FLIGHT:
            g[s:t] = -2 * np.polyfit(np.arange(t - s) / FPS, com_y[s:t], 2)[0]
    return g


def physics_labels(j):
    T = j.shape[0]
    com, contact, feet = center_of_mass(j), foot_contacts(j), j[:, FOOT_JOINTS]
    pts = [feet[t, contact[t]][:, [0, 2]] for t in range(T)]
    margin = np.array([balance_margin(com[t, [0, 2]], pts[t]) for t in range(T)])
    # Extrapolated CoM (Hof 2005): position + velocity / omega0. Unlike the static CoM test,
    # normal walking keeps it near the support, so a negative value is a real balance problem.
    v_com = np.gradient(com, 1.0 / FPS, axis=0)
    omega0 = np.sqrt(9.81 / max(np.median(com[:, 1]), 0.3))
    xcom = com[:, [0, 2]] + v_com[:, [0, 2]] / omega0
    xmargin = np.array([balance_margin(xcom[t], pts[t]) for t in range(T)])
    vel = np.zeros(feet.shape[:2])
    vel[1:] = np.linalg.norm((feet[1:] - feet[:-1])[..., [0, 2]], axis=-1) * FPS
    vel[0] = vel[1]
    skate = np.where(contact.any(1), (vel * contact).sum(1) / np.maximum(contact.sum(1), 1), np.nan)
    # Airborne = every foot joint above its height threshold (pure geometry, so sliding feet don't count)
    # and no knee, wrist or pelvis near the floor (crawling and cartwheels are supported, not in flight).
    airborne = (feet[..., 1] > FOOT_HEIGHT_MAX[None]).all(1) & (j[:, SUPPORT_JOINTS, 1] > SUPPORT_HEIGHT_MIN).all(1)
    return dict(xcom_margin=xmargin, balance_margin=margin, lowest_foot=feet[..., 1].min(1),
                in_flight=(~contact.any(1)).astype(np.float32), skate=skate, contact=contact,
                airborne=airborne.astype(np.float32), flight_gravity=flight_gravity(com[:, 1], airborne))


def labels_for(joints):
    out = [physics_labels(j) for j in joints]
    return {k: np.stack([o[k] for o in out]) for k in out[0]}


t0 = time.time()
LAB = labels_for(J_final)
LAB_X0 = {k: labels_for(X0J[k]) for k in RECORD_STEPS}
print(f"labels computed in {time.time() - t0:.0f}s")
""")

code(r"""
def category(p):
    if any(w in p for w in ["jump", "hop", "leap", "skip"]): return "flight"
    if any(w in p for w in ["one leg", "balanc", "lean", "toes", "tiptoe", "stumble", "lunge", "kick",
                            "tree pose", "squat", "bends down", "stops suddenly", "right foot"]): return "balance"
    return "other"

cats = np.array([category(p) for p in all_prompts])
pred_contact = F_final[..., -4:] > 0.5            # MDM's own foot-contact output channels
agree = (pred_contact == LAB["contact"]).mean()

print(f"{'group':8s} {'motions':>7s} {'flight %':>9s} {'float>5cm %':>12s} {'penetr<-2cm %':>14s} "
      f"{'CoM out %':>10s} {'XCoM out %':>11s} {'skate m/s':>10s} {'airborne %':>11s} {'floating %':>11s} "
      f"{'jump g m/s2':>12s}")
for c in ["flight", "balance", "other"]:
    m = cats == c
    lf, mg, xm, sk = LAB["lowest_foot"][m], LAB["balance_margin"][m], LAB["xcom_margin"][m], LAB["skate"][m]
    ground = LAB["in_flight"][m] == 0
    print(f"{c:8s} {m.sum():7d} {100 * LAB['in_flight'][m].mean():9.1f} {100 * (lf[ground] > 0.05).mean():12.1f} "
          f"{100 * (lf < -0.02).mean():14.1f} {100 * np.mean(mg[np.isfinite(mg)] < 0):10.1f} "
          f"{100 * np.mean(xm[np.isfinite(xm)] < 0):11.1f} {np.nanmedian(sk):10.2f} "
          f"{100 * LAB['airborne'][m].mean():11.1f} {100 * (LAB['flight_gravity'][m] < FLOAT_G_MAX).mean():11.1f} "
          f"{np.nanmedian(np.where(LAB['flight_gravity'][m] >= FLOAT_G_MAX, LAB['flight_gravity'][m], np.nan)):12.2f}")
print(f"\nMDM's own contact channels agree with geometric contact on {100 * agree:.1f}% of foot-frames.")
print("Airborne: no foot, knee, wrist or pelvis near the floor. Over each airborne stretch of 0.2 s or more we fit")
print(f"the fall rate: below {FLOAT_G_MAX} m/s2 it counts as floating; 'jump g' is the median of the rest (real: 9.81).")
print("Note: 'CoM out' and 'XCoM out' also flag normal walking, so read them as descriptive, not as failures.")
""")

code(r"""
# Look at two motions: side-view stick figure. Green dot = CoM inside support, red = outside, blue = in flight.
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
plt.rcParams["animation.embed_limit"] = 30

CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def animate(i, title):
    j, lab = J_final[i], {k: v[i] for k, v in LAB.items()}
    d = j[-1, 0, [0, 2]] - j[0, 0, [0, 2]]
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.3 else np.array([1.0, 0.0])
    u = j[..., 0] * d[0] + j[..., 2] * d[1]          # horizontal axis along travel direction
    com = center_of_mass(j); cu = com[:, 0] * d[0] + com[:, 2] * d[1]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.set_xlim(u.min() - 0.5, u.max() + 0.5); ax.set_ylim(-0.1, 2.1); ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=1); ax.set_title(title, fontsize=8)
    lines = [ax.plot([], [], lw=2, color="k")[0] for _ in CHAIN]
    dot = ax.plot([], [], "o", ms=6)[0]
    def step(t):
        for ln, ch in zip(lines, CHAIN):
            ln.set_data(u[t, ch], j[t, ch, 1])
        dot.set_data([cu[t]], [0.02])
        dot.set_color("tab:blue" if lab["in_flight"][t] else ("tab:green" if lab["balance_margin"][t] >= 0 else "tab:red"))
        return lines + [dot]
    anim = animation.FuncAnimation(fig, step, frames=range(0, N_FRAMES, 2), interval=100)
    plt.close(fig)
    return HTML(anim.to_jshtml())

for p in ["a person jumps forward.", "a person stands on one leg."]:
    display(animate(all_prompts.index(p), p))
""", tag="viz")

md("""
## 5. Probes: when and where is physics readable?

For each recorded denoising step and each layer, a ridge-regression probe predicts the **final** motion's per-frame label from that frame's activation. Train and test sets use **different prompts**, so probes must generalise to new actions.

Baselines at the same step:
- **x_t probe**: linear probe on the noisy input the model sees.
- **x0 probe**: linear probe on the model's current guess of the clean motion (263 features).
- **explicit**: the label computed directly from that x0 guess (what the model "shows").

Scores: R² for balance margin and lowest foot, ROC-AUC for in flight.
""")

code(r"""
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
import pandas as pd

rng = np.random.default_rng(0)
test_prompts = rng.choice(len(PROMPTS), size=len(PROMPTS) // 5, replace=False)
is_test = np.isin(prompt_id, test_prompts)
TARGETS = {"flight_gravity": "r2", "xcom_margin": "r2", "balance_margin": "r2", "lowest_foot": "r2", "in_flight": "auc"}


def flat(a):          # [N, F, ...] -> [N*F, ...]
    return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])

split = flat(np.repeat(is_test[:, None], len(FR), 1))


def score(kind, y, p):
    return roc_auc_score(y, p) if kind == "auc" else r2_score(y, p)


def probe(X, y, ok, kind):
    tr, te = ok & ~split, ok & split
    if tr.sum() < 30 or te.sum() < 10:      # too few frames (e.g. little flight in this batch of motions)
        return np.nan
    if kind == "auc" and len(np.unique(y[te])) < 2:
        return np.nan
    sc = StandardScaler().fit(X[tr].astype(np.float32))
    m = RidgeCV(alphas=np.logspace(0, 4, 5)).fit(sc.transform(X[tr].astype(np.float32)), y[tr])
    return score(kind, y[te], m.predict(sc.transform(X[te].astype(np.float32))))


def explicit(xe, y, ok, kind):
    tr, te = ok & ~split, ok & split
    if tr.sum() < 30 or te.sum() < 10:
        return np.nan
    fill = np.nanmedian(xe[tr]) if np.isfinite(xe[tr]).any() else 0.0
    xe = np.where(np.isfinite(xe), xe, fill)[:, None]
    if kind == "auc":
        return roc_auc_score(y[te], xe[te, 0]) if len(np.unique(y[te])) > 1 else np.nan
    m = LinearRegression().fit(xe[tr], y[tr])
    return r2_score(y[te], m.predict(xe[te]))


rows = []
t0 = time.time()
for name, kind in TARGETS.items():
    y = flat(LAB[name][:, FR])
    ok = np.isfinite(y)
    for k in RECORD_STEPS:
        rows.append(dict(target=name, step=k, source="x_t probe", score=probe(flat(XT[k]), y, ok, kind)))
        rows.append(dict(target=name, step=k, source="x0 probe", score=probe(flat(X0F[k]), y, ok, kind)))
        rows.append(dict(target=name, step=k, source="explicit", score=explicit(flat(LAB_X0[k][name][:, FR]), y, ok, kind)))
        for li in range(len(LAYERS)):
            rows.append(dict(target=name, step=k, source=f"layer {li}", score=probe(flat(ACTS[(k, li)]), y, ok, kind)))
    print(f"{name}: done ({time.time() - t0:.0f}s)")
RES = pd.DataFrame(rows)
""")

code(r"""
import matplotlib.pyplot as plt
from IPython.display import display

for name, kind in TARGETS.items():
    d = RES[RES.target == name]
    layer_tab = d[d.source.str.startswith("layer")].pivot(index="source", columns="step", values="score")
    layer_tab = layer_tab.loc[[f"layer {i}" for i in range(len(LAYERS))]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=90)
    lo = 0.5 if kind == "auc" else 0.0
    im = axes[0].imshow(layer_tab.values, aspect="auto", cmap="viridis", vmin=lo, vmax=1)
    axes[0].set_xticks(range(len(RECORD_STEPS)), RECORD_STEPS); axes[0].set_yticks(range(len(LAYERS)), layer_tab.index)
    axes[0].set_xlabel("denoising step (0 = noise)"); axes[0].set_title(f"{name}: layer probes ({kind.upper()})")
    for (r, c), v in np.ndenumerate(layer_tab.values):
        axes[0].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
    plt.colorbar(im, ax=axes[0])
    best = layer_tab.max(0)
    axes[1].plot(RECORD_STEPS, best.values, "o-", label="best layer", lw=2)
    for src in ["x0 probe", "explicit", "x_t probe"]:
        s = d[d.source == src].set_index("step").score.reindex(RECORD_STEPS)
        axes[1].plot(RECORD_STEPS, s.values, "o--", label=src)
    axes[1].set_ylim(lo - 0.1, 1.02); axes[1].set_xlabel("denoising step (0 = noise)")
    axes[1].set_title("best layer vs baselines"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

summary = RES.pivot_table(index=["target", "source"], columns="step", values="score").round(3)
display(summary)
""")

md("""
## 6. Early failure prediction (the practical payoff)

Can we tell, partway through denoising, which samples **will end up breaking physics**? Each motion gets four scores from its final output:
- **gravity error (jumps)** (main): average gap between the fitted fall rate and 9.81 m/s² over unsupported airborne stretches that do fall (at least 2 m/s²). Only motions with a jump have it.
- **floating rate**: share of frames in unsupported airborne stretches that barely fall (under 2 m/s²): off the ground, but not dropping.
- **float rate**: share of ground frames where the lowest foot is more than 5 cm up.
- **off-balance rate (exploratory)**: share of support frames where the extrapolated CoM is outside the support area. Real people also score high on this during normal stepping, so treat it as descriptive until we calibrate it on real motion.

For each step and layer, a ridge probe on frame-averaged activations predicts these scores. Folds are split **by prompt**, so the probe never sees the test prompts.

The key control is **text only**: a probe on the prompt's CLIP embedding. Some prompts (jumps) simply float more, so beating the text baseline is what matters. The **within-prompt** score goes further: among samples of the *same* prompt, does the probe rank the bad ones higher? That can only come from the sample itself, not the prompt.

Note: "best layer" picks the top of 8 layers, which slightly flatters it. If the effect looks real, confirm it with one fixed layer chosen in advance.
""")

code(r"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr


def failure_rates(lab):
    ground = lab["in_flight"] == 0
    flt = np.where(ground, lab["lowest_foot"] > 0.05, np.nan).astype(float)
    xm = lab["xcom_margin"]
    bal = np.where(np.isfinite(xm), xm < 0, np.nan).astype(float)
    g = lab["flight_gravity"]
    # Unsupported stretches split in two: jumps (judged by how far their fall rate is from 9.81)
    # and floating (the body is off the ground but barely falls).
    jump_err = np.where(g >= FLOAT_G_MAX, np.abs(g - G_EARTH), np.nan)
    floating = (np.isfinite(g) & (g < FLOAT_G_MAX)).astype(float)
    with np.errstate(all="ignore"):
        return {"gravity error (jumps)": np.nanmean(jump_err, 1), "floating rate": floating.mean(1),
                "float rate": np.nanmean(flt, 1), "off-balance rate (exploratory)": np.nanmean(bal, 1)}


FAIL = failure_rates(LAB)
FAIL_X0 = {k: failure_rates(LAB_X0[k]) for k in RECORD_STEPS}
gkf = GroupKFold(n_splits=5)


def within_prompt(pred, y, ok):
    # Spearman correlation after removing each prompt's mean: ranks samples *within* a prompt.
    rp, ry, keep = pred.copy(), y.copy(), np.zeros(len(y), bool)
    for pid in np.unique(prompt_id[ok]):
        m = ok & (prompt_id == pid)
        if m.sum() >= 2:
            rp[m] -= rp[m].mean(); ry[m] -= ry[m].mean(); keep |= m
    return spearmanr(rp[keep], ry[keep]).correlation if keep.sum() >= 10 else np.nan


def oof_ridge(X, y, ok, alpha=10.0):
    # Out-of-fold predictions with prompt-grouped folds; fits only on samples where the outcome exists.
    X = X.astype(np.float32)
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups=prompt_id):
        tr = tr[ok[tr]]
        if len(tr) < 10:
            pred[te] = np.nan
            continue
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pred


rows = []
for fname, yv in FAIL.items():
    ok = np.isfinite(yv)
    y = np.where(ok, yv, np.nanmean(yv))
    def add(step, source, pred):
        good = ok & np.isfinite(pred)
        rows.append(dict(failure=fname, step=step, source=source, n=int(ok.sum()),
                         spearman=spearmanr(pred[good], y[good]).correlation if good.sum() >= 10 else np.nan,
                         within_prompt=within_prompt(pred, y, good)))
    add(-1, "text only", oof_ridge(TXT, y, ok, alpha=100.0))
    for k in RECORD_STEPS:
        e = FAIL_X0[k][fname]
        add(k, "explicit (x0 guess)", np.where(np.isfinite(e), e, np.nanmean(e)))
        add(k, "x0 probe", oof_ridge(X0F[k].astype(np.float32).mean(1), y, ok))
        cands = [oof_ridge(ACTS[(k, li)].astype(np.float32).mean(1), y, ok) for li in range(len(LAYERS))]
        best = max(cands, key=lambda pr: np.nan_to_num(spearmanr(pr[ok & np.isfinite(pr)], y[ok & np.isfinite(pr)]).correlation, nan=-1))
        add(k, "best layer", best)
EARLY = pd.DataFrame(rows)

fig, axes = plt.subplots(1, len(FAIL), figsize=(5.5 * len(FAIL), 3.6), dpi=90)
for ax, fname in zip(axes, FAIL):
    d = EARLY[EARLY.failure == fname]
    txt = d[d.source == "text only"].spearman.values[0]
    for src in ["best layer", "x0 probe", "explicit (x0 guess)"]:
        s = d[d.source == src].set_index("step")
        ax.plot(RECORD_STEPS, s.spearman.reindex(RECORD_STEPS), "o-", label=f"{src}")
        if src == "best layer":
            ax.plot(RECORD_STEPS, s.within_prompt.reindex(RECORD_STEPS), "o:", label="best layer, within prompt")
    ax.axhline(txt, color="gray", ls="--", label="text only")
    ax.set_title(f"predicting final {fname}"); ax.set_xlabel("denoising step (0 = noise)")
    ax.set_ylabel("Spearman ρ"); ax.set_ylim(-0.3, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
display(EARLY.pivot_table(index=["failure", "source"], columns="step", values=["spearman", "within_prompt"]).round(3))
""")

code(r"""
# Save compact results (labels, probe scores, prompts) so nothing is lost if the runtime resets.
np.savez_compressed("/content/balance_probe_results.npz",
                    joints=J_final.astype(np.float32), prompts=np.array(all_prompts),
                    **{f"label_{k}": v for k, v in LAB.items()})
RES.to_csv("/content/probe_scores.csv", index=False)
EARLY.to_csv("/content/early_prediction.csv", index=False)
print("Saved balance_probe_results.npz, probe_scores.csv and early_prediction.csv in /content (Files panel).")
""", tag="save")

md("""
## 7. How to read the results

- **Layer probes beat "explicit" at early or middle steps** → the model represents the final motion's balance or flight *before* its own clean-motion guess shows it. This is the "knows more than it shows" result and the core of the paper.
- **Layer probes only match "x0 probe"** → the physics is no richer inside than in the output guess. Still useful, but the story becomes about decoding, not hidden knowledge.
- **Low scores everywhere for balance margin but high for flight** → the model tracks contact and flight but not balance. That's a clean, publishable negative result for balance.
- **Early prediction beats "text only", and "within prompt" stays clearly above 0 at early steps** → the model knows *this sample* is going wrong before it finishes. That's the early-warning result: resample bad samples early and save compute.
- **Physics table above**: high "CoM outside support %" in the balance group, or high floating in the flight group, tells us which artifact to focus on.

**Next steps after this test**
1. Causal check: add or subtract the probe direction during sampling and measure whether balance in the output changes.
2. Early-failure detector: from step-k activations, predict which samples will float or lose balance, then resample them early.
3. Repeat with a second architecture (the MDM decoder model, or MoMask) to test generality.
""")


def build(path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    for kind, tag, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, path)


def export_test(path, skip=("setup",)):
    with open(path, "w") as f:
        for kind, tag, src in cells:
            if kind == "code" and tag not in skip:
                f.write(src.replace("from IPython.display import display", "display = print") + "\n\n")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    build(os.path.join(here, "..", "notebooks", "motion_balance_probe.ipynb"))
    print("built")
