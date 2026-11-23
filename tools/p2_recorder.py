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