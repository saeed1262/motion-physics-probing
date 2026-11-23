"""Encode the IntPhys dev videos with the Wan 2.1 VAE (clean latents only, no diffusion model).

Preprocessing follows Esmati et al.'s IntPhys setting as far as their public code shows: RGB
frames resized to 256 x 256, 100 frames requested (the Wan VAE consumes 1 + 4k frames, so it
uses the first 97 and returns 25 latent frames), pixel values scaled to [-1, 1]. The latent is
the posterior mean, shape [16 channels, 25 frames, 32, 32].

    .venv/bin/python scripts/intphys_vae_encode.py --check     # Apple GPU vs CPU on 3 videos, plus a reconstruction
    .venv/bin/python scripts/intphys_vae_encode.py             # encode all 360 dev videos

Writes data/intphys_vae/latents.npz (float16 latents, labels, block, scenario, condition).
"""
import argparse, json, os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKLWan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "vendor", "intphys", "dev")
OUT = os.path.join(ROOT, "data", "intphys_vae")
ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
CLI = ap.parse_args()
N_FRAMES, SIZE = 97, 256


def scenes():
    out = []
    for block in sorted(os.listdir(DEV)):
        if not block.startswith("O"):
            continue
        for scen in sorted(os.listdir(os.path.join(DEV, block))):
            for cond in sorted(os.listdir(os.path.join(DEV, block, scen))):
                d = os.path.join(DEV, block, scen, cond)
                if os.path.isfile(os.path.join(d, "status.json")):
                    out.append((block, scen, cond, d))
    return out


def load_video(d):
    frames = [np.asarray(Image.open(os.path.join(d, "scene", f"scene_{i:03d}.png")).convert("RGB"))
              for i in range(1, N_FRAMES + 1)]
    x = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2) / 255.0      # [3, T, H, W]
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    return (x * 2 - 1)[None]                                                         # [1, 3, T, 256, 256]


def load_vae(device):
    vae = AutoencoderKLWan.from_pretrained(os.path.join(ROOT, "vendor", "wan_vae", "vae"), torch_dtype=torch.float32)
    return vae.to(device).eval()


@torch.no_grad()
def encode(vae, x, device):
    return vae.encode(x.to(device)).latent_dist.mean.float().cpu()


all_scenes = scenes()
print(f"{len(all_scenes)} videos; device {CLI.device}")
if CLI.check:
    vae_cpu, vae_dev = load_vae("cpu"), load_vae(CLI.device)
    for block, scen, cond, d in all_scenes[:3]:
        x = load_video(d)
        t0 = time.time(); zc = encode(vae_cpu, x, "cpu"); tc = time.time() - t0
        t0 = time.time(); zd = encode(vae_dev, x, CLI.device); td = time.time() - t0
        rel = ((zd - zc).norm() / zc.norm()).item()
        per_frame = ((zd - zc).flatten(3).norm(dim=-1) / zc.flatten(3).norm(dim=-1)).max().item()
        print(f"{block}/{scen}/{cond}: latent {tuple(zc.shape)} | {CLI.device} vs cpu rel diff {rel:.2e} "
              f"(worst latent frame/channel {per_frame:.2e}) | cpu {tc:.1f}s, {CLI.device} {td:.1f}s")
    with torch.no_grad():
        rec = vae_dev.decode(zd.to(CLI.device)).sample.float().cpu().clamp(-1, 1)
    mse = ((rec[:, :, :x.shape[2]] - x) ** 2).mean(dim=(0, 1, 3, 4))   # per frame, in [-1, 1] units
    psnr = 10 * torch.log10(4.0 / mse)
    print(f"reconstruction on {CLI.device}: PSNR per frame min {psnr.min():.1f} dB, median {psnr.median():.1f} dB")
else:
    os.makedirs(OUT, exist_ok=True)
    vae = load_vae(CLI.device)
    Z, meta, t0 = [], [], time.time()
    for i, (block, scen, cond, d) in enumerate(all_scenes):
        Z.append(encode(vae, load_video(d), CLI.device)[0].numpy().astype(np.float16))
        meta.append(dict(block=block, scenario=f"{block}/{scen}", condition=cond,
                         label=int(json.load(open(os.path.join(d, "status.json")))["header"]["is_possible"])))
        if (i + 1) % 30 == 0:
            print(f"{i + 1}/{len(all_scenes)} encoded ({time.time() - t0:.0f}s)", flush=True)
    Z = np.stack(Z)
    assert np.isfinite(Z.astype(np.float32)).all()
    np.savez(os.path.join(OUT, "latents.npz"), z=Z, label=np.array([m["label"] for m in meta]),
             block=np.array([m["block"] for m in meta]), scenario=np.array([m["scenario"] for m in meta]),
             condition=np.array([m["condition"] for m in meta]))
    print("saved", Z.shape, "in", OUT, f"({time.time() - t0:.0f}s)")
