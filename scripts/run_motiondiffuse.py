"""Record MotionDiffuse (Zhang et al. 2022) along its own sampling trajectories, in the same format as
the MDM runs, so every analysis script reads it unchanged (checkpoint name "motiondiffuse").

MotionDiffuse predicts the noise (epsilon), so its clean-motion estimate is computed from the prediction
and the noisy input; it uses no classifier-free guidance; its 8 layers keep one token per frame, and the
text enters every layer by cross-attention and through the stylization blocks. Sampling is the official
one: DDPM, 1000 steps, linear betas, fixed small variance, clip_denoised=False.

Same as the MDM runs: the prompts of the matching MDM run (data/<tag>/50steps_seed0/meta.json), 4
samples per prompt, batches of 64, noise seed seed * 100003 + batch, 120 frames, recorded steps
0/200/400/599/799/999, every 4th frame kept for activations and features, labels from
tools/p2_labels_funcs.py. Files: data/<tag>/motiondiffuse_seed<s>/{meta.json, batch_XXX.npz, labels.npz}.

    .venv/bin/python scripts/run_motiondiffuse.py --tag phase2 --seeds 0,1,2 [--device mps] [--smoke]

Needs vendor/motiondiffuse (github.com/MotrixLab/MotionDiffuse) and its HumanML3D checkpoint in
vendor/motiondiffuse/text2motion/checkpoints/t2m/t2m_motiondiffuse/ (from the official Hugging Face
Space mingyuan/MotionDiffuse). mmcv is not needed: inference never imports the training code.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "vendor", "motiondiffuse", "text2motion")
CK = os.path.join(MD, "checkpoints", "t2m", "t2m_motiondiffuse")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="phase2", help="prompt set and output folder: data/<tag>/")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--samples-per-prompt", type=int, default=4)
ap.add_argument("--smoke", action="store_true", help="2 prompts x 2 samples, into data/<tag>_smoke/")
CLI = ap.parse_args()
DEV = torch.device(CLI.device)
N_FRAMES, FRAME_STRIDE, BATCH, N_RECORD = 120, 4, 64, 6
FR = np.arange(0, N_FRAMES, FRAME_STRIDE)

sys.path.insert(0, MD)
import types                                                                     # noqa: E402
sys.modules.setdefault("cv2", types.SimpleNamespace(norm=None))   # models/transformer.py imports cv2.norm and never uses it
import clip                                                                    # noqa: E402
import models.gaussian_diffusion as gd                                           # noqa: E402
from models.transformer import MotionTransformer                                 # noqa: E402
sys.path.append(os.path.join(ROOT, "vendor", "mdm"))
# MDM's copy: the function is identical to MotionDiffuse's, whose quaternion helpers use np.float (gone from NumPy).
from data_loaders.humanml.scripts.motion_process import recover_from_ric         # noqa: E402


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    # MPS has no float64: index on the CPU and cast first (same float32 values on any device).
    res = torch.from_numpy(arr)[timesteps.cpu()].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


gd._extract_into_tensor = _extract_into_tensor

# The official model and diffusion settings (utils/get_opt.py defaults, trainers/ddpm_trainer.py).
model = MotionTransformer(input_feats=263, num_frames=196, num_layers=8, latent_dim=512, no_clip=False, no_eff=False)
ckpt = torch.load(os.path.join(CK, "model", "latest.tar"), map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["encoder"], strict=True)
model.to(DEV).eval()
diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule("linear", 1000), model_mean_type=gd.ModelMeanType.EPSILON,
                                 model_var_type=gd.ModelVarType.FIXED_SMALL, loss_type=gd.LossType.MSE)
MEAN = torch.tensor(np.load(os.path.join(CK, "meta", "mean.npy"))).float()
STD = torch.tensor(np.load(os.path.join(CK, "meta", "std.npy"))).float()
N_STEPS = diffusion.num_timesteps
RECORD_STEPS = sorted(set(np.linspace(0, N_STEPS - 1, N_RECORD).round().astype(int).tolist()))
LAYERS = list(model.temporal_decoder_blocks)
print(f"MotionDiffuse epoch {ckpt.get('ep')} | {len(LAYERS)} layers | {N_STEPS} steps | recording {RECORD_STEPS} | {DEV}", flush=True)
del ckpt


class Recorder:
    # Forward hooks on every decoder layer; every token is a frame (no condition token to drop).
    def __init__(self, layers):
        self.on, self.buf = False, {}
        for li, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(li))

    def _hook(self, li):
        def hook(module, inputs, out):
            if self.on:
                self.buf[li] = out[:, ::FRAME_STRIDE].to(torch.float16).cpu()   # [bs, F, d]
        return hook


recorder = Recorder(LAYERS)


def to_joints(x):
    # Normalised output [bs, T, 263] -> joints [bs, T, 22, 3] (metres, Y up) and raw features.
    v = x.float().cpu() * STD + MEAN
    return recover_from_ric(v, 22).numpy(), v.numpy()


@torch.no_grad()
def sample_batch(prompts, seed):
    bs = len(prompts)
    torch.manual_seed(seed)
    xf_proj, xf_out = model.encode_text(list(prompts), DEV)
    # CLIP's pooled text embedding, the same kind of `txt` the MDM runs store (for the text control).
    txt = model.clip.encode_text(clip.tokenize(list(prompts), truncate=True).to(DEV)).float().cpu().numpy()
    kw = {"xf_proj": xf_proj, "xf_out": xf_out, "length": torch.full((bs,), N_FRAMES, dtype=torch.long)}
    x = torch.randn((bs, N_FRAMES, 263), device=DEV)
    acts, xt_feats, x0 = {}, {}, {}
    for k, t_int in enumerate(range(N_STEPS - 1, -1, -1)):
        t = torch.full((bs,), t_int, device=DEV, dtype=torch.long)
        recorder.on, recorder.buf = k in RECORD_STEPS, {}
        out = diffusion.p_sample(model, x, t, clip_denoised=False, model_kwargs=kw)
        recorder.on = False
        if k in RECORD_STEPS:
            for li, h in recorder.buf.items():
                acts[(k, li)] = h.numpy()
            xt_feats[k] = x[:, ::FRAME_STRIDE].to(torch.float16).cpu().numpy()
            x0[k] = to_joints(out["pred_xstart"])     # the clean estimate, computed from the predicted noise
        x = out["sample"]
    joints, feats = to_joints(x)
    return joints, feats, acts, xt_feats, x0, txt


LAB = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), LAB)

src = json.load(open(os.path.join(ROOT, "data", CLI.tag, "50steps_seed0", "meta.json")))
PROMPTS = src["prompts"]
if CLI.smoke:
    PROMPTS, CLI.samples_per_prompt = PROMPTS[8:10], 2
all_prompts = [p for _ in range(CLI.samples_per_prompt) for p in PROMPTS]
prompt_id = [i for _ in range(CLI.samples_per_prompt) for i in range(len(PROMPTS))]
OUT_TAG = CLI.tag + ("_smoke" if CLI.smoke else "")

for seed in [int(s) for s in CLI.seeds.split(",")]:
    d = os.path.join(ROOT, "data", OUT_TAG, f"motiondiffuse_seed{seed}")
    os.makedirs(d, exist_ok=True)
    meta = dict(checkpoint="motiondiffuse", seed=seed, prompts=PROMPTS, all_prompts=all_prompts, prompt_id=prompt_id,
                record_steps=RECORD_STEPS, frame_stride=FRAME_STRIDE, n_frames=N_FRAMES, batch=BATCH,
                n_layers=len(LAYERS), guidance=None)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    n_batches, t0 = -(-len(all_prompts) // BATCH), time.time()
    for b in range(n_batches):
        path = os.path.join(d, f"batch_{b:03d}.npz")
        if os.path.exists(path):
            continue
        j, f, acts, xt, x0, txt = sample_batch(all_prompts[b * BATCH:(b + 1) * BATCH], seed=seed * 100003 + b)
        arrs = dict(joints=j.astype(np.float32), feats=f[:, FR].astype(np.float16), txt=txt.astype(np.float16))
        for (k, li), v in acts.items():
            arrs[f"act_{k}_{li}"] = v
        for k in RECORD_STEPS:
            arrs[f"xt_{k}"] = xt[k]
            arrs[f"x0j_{k}"] = x0[k][0].astype(np.float16)
            arrs[f"x0f_{k}"] = x0[k][1][:, FR].astype(np.float16)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrs)
        os.replace(tmp, path)    # only complete batches ever appear under the final name
        print(f"seed {seed}: batch {b + 1}/{n_batches} saved ({time.time() - t0:.0f}s)", flush=True)
    lab_path = os.path.join(d, "labels.npz")
    if not os.path.exists(lab_path):
        files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
        load = lambda key: np.concatenate([np.load(p)[key] for p in files])
        final = LAB["labels_for"](load("joints"))
        x0l = {k: LAB["labels_for"](load(f"x0j_{k}").astype(np.float32)) for k in RECORD_STEPS}
        np.savez(lab_path, final=np.array(final, dtype=object), x0=np.array(x0l, dtype=object))
    print(f"seed {seed}: done, labels in {lab_path} ({time.time() - t0:.0f}s)", flush=True)
