# Findings log

Results of the runs on the local Mac (Apple M4 Pro, MPS), November 15 to 16, 2026. Every number
here was produced by the scripts listed at the end; the CSVs hold the full tables.
Interpretations are marked as such. Read with `RESEARCH_PLAN.md` (hypotheses).

**Label version.** All numbers below use the airborne/gravity labels as fixed on November 16
(section 6). Results under the original labels are kept in `results/archive_labels_v1/`. The
fix changed only the gravity-related labels; every other probe score is identical to the
original runs (sampling on MPS reproduced exactly).

## Current verdict

| Hypothesis | Status | Evidence |
| --- | --- | --- |
| H1: physics is linearly readable from MDM's layers | **Supported** | Dynamic balance R² 0.84, flight AUC 0.97, lowest foot R² 0.87, jump gravity R² 0.82 at 20% of denoising (50-step model, 3 seeds, test captions included) |
| H2: early activations predict final physics better than the model's own clean-motion guess | **Not supported** under a fair readout | Balance (dynamic and static) and flight: a small MLP on what the guess shows matches or beats the layer, and adding the layer adds nothing, pooled or within prompts. Gravity: the layer helps on the 50-step model's pooled score only; no gain on the 1000-step model, and within prompts the sign flips between seeds (section 4) |
| H3: early activations flag failing samples beyond the prompt | **Weak; the guess does it better** | Within-prompt ρ at 20% for the four failure measures: fixed layer −0.09 to 0.52, explicit guess 0.10 to 0.71 (section 3). As an abort-and-resample rule the guess catches 45 to 65% of floating samples at 80 to 88% precision, but saves only 3 to 9% of compute (section 7) |

Interpretation: the paper as planned ("knows more than it shows") does not hold once the guess
is read out fairly. The closest video papers make that claim without this kind of control
(section 5), so the controls themselves become the contribution. The same pitfall shows up in a
video paper: plausibility that a fully pooled linear probe cannot see in the clean VAE latent
becomes readable (80% pairwise) once the readout keeps frame-to-frame change (section 8). RESEARCH_PLAN's fallback
applies: "probes only match the clean-motion-guess baseline, so the story shifts to how physics
is decoded across steps." Still a first internal analysis of a human motion generator, with a
practical early-warning lesson (decode the guess).

## 1. Setup and code changes

- Local scripts are generated from the notebook sources by `tools/build_local.py`:
  `scripts/run_phase1_local.py` and `scripts/run_phase2_local.py`. Labels, probes, baselines and
  thresholds are copied verbatim (the generator asserts it).
- Mac-only shim: MDM moves float64 noise-schedule tables to the device and MPS has no float64;
  they are cast to float32 on the CPU first (same values). A single forward pass on MPS matches
  CPU to about 1e-4 relative error on every layer.
- Two notebook bugs fixed in `tools/build_nb*.py` (both would also crash on Colab):
  - the text-only baseline: `encode_text` returns `[1, batch, 512]`, so per-sample text
    features did not line up with motions (phase 1 and phase 2);
  - phase 2 model loading: `model.to(DEVICE).eval()` fails because MDM's `.to()` returns None.
- Label fix (approved November 16; section 6), in `tools/build_nb.py` and so in both notebooks
  and both scripts.
- Not changed: the Colab notebooks still call `gdown --fuzzy`, which gdown 6 removed. Colab's
  preinstalled gdown may be older.
- MDM pinned to commit `ef8edce`. Checkpoints: 50-step `model000750000.pt`, 1000-step
  `model000475000.pt`. EMA weights not used (`use_ema` is false in the 50-step args).

## 2. Phase 1 (64 prompts x 4 samples, one held-out set of 12 prompts)

Physics table (256 motions):

| Group | Motions | Flight % | Float >5 cm % | XCoM out % | Airborne % | Floating % | Jump gravity, median (m/s²) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flight | 56 | 16.1 | 0.9 | 27.8 | 12.6 | 0.6 | 9.75 |
| balance | 68 | 9.1 | 2.4 | 30.1 | 4.8 | 4.5 | 7.48 |
| other | 132 | 8.7 | 2.4 | 35.0 | 5.1 | 4.4 | 7.14 |

- 66 of 256 motions have an unsupported airborne stretch of at least 0.2 s (41 flight, 9
  balance, 16 other).
- Probes at step 10 (20%), best layer vs explicit: dynamic balance 0.814 vs 0.832, flight AUC
  0.977 vs 0.940, flight gravity 0.828 vs 0.553.
- Call at the time: conditional go (through flight, by small margins).

## 3. Phase 2 (250 prompts x 4 samples, 5-fold CV by prompt)

Runs: 50-step model seeds 0, 1, 2; original 1000-step model seeds 0, 1. 1,000 motions per run.

**Physics.** When MDM's bodies actually fall, they fall at about the right rate; the
difference between checkpoints is floating:

| Measure | 50-step (3 seeds) | 1000-step (2 seeds) |
| --- | --- | --- |
| Jump gravity, curated jump prompts (median, m/s²) | 9.54 to 9.75 | 9.13 to 9.18 |
| Jump gravity, test captions (median, m/s²) | 9.07 to 9.84 | 8.59 to 8.76 |
| Floating % of frames, test captions | 3.8 to 4.7 | 8.6 to 8.9 |
| Floating % of frames, curated jump prompts | 0.6 to 0.8 | 5.2 to 9.3 |
| Float >5 cm % of contact frames, curated jump prompts | 1.2 to 2.4 | 5.8 to 8.4 |

- Interpretation: the original 1000-step model floats about twice as often as the 50-step one.
  Under the original labels this looked like "its jumps fall too slowly" (median 3.75 and
  7.55 m/s²); that was floating mixed into the jump fits.

**Probe scores at 20% of denoising** (mean ± spread across seeds); "gap" = best layer minus explicit:

| Target | 50-step best layer | 50-step explicit | Gap (50-step) | Gap (1000-step) |
| --- | --- | --- | --- | --- |
| Dynamic balance (XCoM margin) | 0.838 ± 0.001 | 0.783 ± 0.009 | +0.045 to +0.065 per seed | +0.048, +0.060 per seed |
| Static balance | 0.759 ± 0.014 | 0.769 ± 0.019 | −0.013 to −0.005 per seed | +0.021, +0.049 per seed |
| Lowest foot | 0.868 ± 0.022 | 0.921 ± 0.005 | −0.073 to −0.039 per seed | −0.053, −0.044 per seed |
| In flight (AUC) | 0.973 ± 0.002 | 0.876 ± 0.005 | +0.094 to +0.105 per seed | +0.134, +0.141 per seed |
| Flight gravity | 0.820 (mean) | 0.707 (mean) | +0.113 (mean) | 0.000 (mean; −0.044 at 40%) |

- Layer 7 was the best layer for dynamic balance at 20% in all five runs.
- Curated prompts vs test captions: same direction for every target (dynamic balance at step
  10: 0.835 vs 0.743 curated, 0.839 vs 0.797 test captions).
- The in-flight comparison favours probes because the explicit score is only 0 or 1.

**Early warning** (layer fixed on seed 0: layer 3 for 50-step, layer 6 for 1000-step), within-prompt
Spearman ρ at 20%, mean over runs:

| Failure | 50-step layer | 50-step explicit | 1000-step layer | 1000-step explicit |
| --- | --- | --- | --- | --- |
| Gravity error, jumps only (about 100 to 125 motions per run) | −0.09 | 0.10 | 0.14 | 0.18 |
| Floating rate | 0.28 | 0.52 | 0.27 | 0.37 |
| Float rate (contact frames) | 0.11 | 0.30 | 0.24 | 0.30 |
| Off-balance (exploratory) | 0.52 | 0.71 | 0.52 | 0.64 |

## 4. Controls (`scripts/phase2_controls.py`)

**Fair readout of the guess.** Same folds; the same small MLP (2 x 256) for every feature set.
"Everything the guess shows" = the guess's 263 features plus every physics label computed from
the guess. Layer fixed as the best mean layer at 20% for each target (dynamic and static balance
7, flight 6, gravity 6 for 50-step and 3 for 1000-step). Scores at 20% of denoising, mean over runs:

| Target | Model | Explicit | Guess features, linear (x0 probe) | Everything the guess shows, MLP | Layer, linear | Shown + layer, MLP | Adding the layer, per run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic balance (R²) | 50-step | 0.783 | 0.723 | **0.878** | 0.838 | 0.867 | −0.019 to −0.006 |
| Dynamic balance (R²) | 1000-step | 0.659 | 0.565 | 0.754 | 0.713 | **0.759** | −0.008 to +0.005 |
| Static balance (R²) | 50-step | 0.769 | 0.673 | **0.816** | 0.759 | 0.791 | −0.046 to −0.007 |
| Static balance (R²) | 1000-step | 0.556 | 0.542 | **0.642** | 0.591 | 0.633 | −0.019 to −0.002 |
| Flight (AUC) | 50-step | 0.876 | 0.935 | **0.972** | **0.972** | 0.967 | −0.012 to +0.005 |
| Flight (AUC) | 1000-step | 0.787 | 0.892 | 0.911 | **0.925** | 0.920 | −0.004 to +0.012 |
| Flight gravity (R²) | 50-step | 0.707 | 0.547 | 0.703 | 0.802 | **0.815** | +0.049 to +0.146 |
| Flight gravity (R²) | 1000-step | 0.692 | 0.330 | 0.633 | 0.681 | **0.711** | +0.033 to +0.111 |

Within prompts (prompt means subtracted from target and prediction; Pearson r), at 20%:

| Target | Model | Explicit | Everything the guess shows, MLP | Layer, MLP | Shown + layer, MLP |
| --- | --- | --- | --- | --- | --- |
| Dynamic balance | 50-step | 0.797 | **0.892** | 0.867 | 0.889 |
| Dynamic balance | 1000-step | 0.738 | 0.816 | 0.803 | **0.819** |
| Static balance | 50-step | 0.746 | **0.801** | 0.749 | 0.795 |
| Flight | 50-step | 0.710 | 0.755 | 0.734 | **0.760** |
| Flight gravity | 50-step | 0.439 | 0.340 | **0.530** | 0.488 |
| Flight gravity | 1000-step | **0.622** | 0.570 | 0.602 | 0.617 |

- Balance and flight: the guess, read out fairly, holds everything the layer holds, pooled and
  within prompts. Interpretation: no detectable hidden knowledge.
- Gravity: the layer beats the guess on the 50-step model's pooled score, but not on the
  1000-step model. Within prompts, layer minus explicit is −0.089, +0.174, +0.189 (seeds 0, 1,
  2) at 20% and −0.157, +0.050, +0.125 at 40% for the 50-step model, and +0.044, −0.084 at 20%
  for the 1000-step model. Interpretation: inconclusive. Gravity has the least data (1,484 to
  1,661 labelled frames per 50-step run, about 3,000 per 1000-step run), and the MLP on "everything the guess shows"
  underuses the explicit gravity feature, so explicit is the stronger guess-side baseline here.
- A 512-dim CLIP prompt embedding added to the readouts made them worse (it can only memorize
  prompts that never appear in the test fold), so it is not a usable control. (Run on the
  original labels; `results/archive_labels_v1/phase2/controls/*_with_text_*`.)

**Why phase 1 and phase 2 disagreed** (dynamic balance, 50-step data, step 10, same test folds,
training prompts subsampled, 3 repeats):

| Training prompts | 13 | 26 | 52 | 100 | all (~200) |
| --- | --- | --- | --- | --- | --- |
| Explicit | 0.780 | 0.782 | 0.783 | 0.783 | 0.779 to 0.792 |
| Layer 7 (linear) | 0.553 | 0.646 | 0.744 | 0.798 | 0.837 to 0.839 |

- The 512-feature probe needs about 100 training prompts to pass the one-number explicit
  baseline; phase 1 trained on 52. Phase 1's exact split applied to phase 2 data flips sign
  between seeds (gap −0.003, +0.022, +0.049 at step 10), so its 12-prompt test set was also noisy.

## 5. Literature check on baselines (November 16, 2026)

Question: do the closest papers compare internal probes against a fair readout of the model's
own clean-output prediction? Read via arXiv: Esmati et al. full HTML text (v2, main text and
appendices); Kneifl et al. PDF pages 1 to 9 (appendices not read); Tang et al. and Joseph et al.
HTML (about the first half verbatim, the rest as a machine summary).

| Paper | Activations from | Baselines for the probes | Readout of the model's own clean prediction? |
| --- | --- | --- | --- |
| Esmati et al., "The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show" (arXiv 2606.05328; listed as a NeurIPS 2026 poster) | Real videos inverted to noise | Linear probes on VAE latents (chance), V-JEPA 2, VideoMAE | No. Linear probes only (one per block on mean-pooled tokens, single 60/40 split, App. F). "Constructed by the model itself" rests on a linear probe on the clean VAE latent at chance plus t-SNE (Sec. 4.1, App. C) |
| Kneifl et al., "Does Physics Live in the Activations?" (arXiv 2610.03154) | Forward-noised simulated videos | Ridge on the noisy input latent; clean latent as a reference (linear); permutation null | No; claim is "beyond the input". Gravity: clean latent R² 0.07 to 0.12 per frame vs activations 0.41 to 0.81, all linear |
| Tang et al., "Seeking Physics in Diffusion Noise" (arXiv 2603.14294) | VideoPhy videos forward-noised; own trajectories at inference | Early pruning vs random selection, best-of-N, DPO, LikePhys | No; argues pixel-space verifiers fail on noisy samples, but does not test x0-preview scoring |
| Joseph et al., "Interpreting Physics in Video World Models" (arXiv 2602.07050) | Encoder models (V-JEPA 2, VideoMAE), no diffusion | Linear and attentive-MLP probes, control tasks, random subspaces | Not applicable |
| Punzo et al. (arXiv 2606.09646) | V-JEPA, VideoMAE, LTX-Video | Layerwise probes, frame shuffling | Not checked in detail |

- No paper found that probes a human motion generator for physics (search on November 16, 2026).
- Image diffusion has early-quality probes (Probe-Select, CVPR 2026; Diffusion Probe, arXiv
  2602.23783); from search snippets their baselines are random selection, not x0-preview
  scoring. Not read in full.
- Interpretation: none of the diffusion papers tests "knows more than it shows" against a
  capacity-matched readout of the model's own prediction, within prompts, with learning
  curves. That is the methodological gap this project's controls fill.

## 6. The label fix: near-zero-gravity ("hovering") stretches (`scripts/inspect_hovering.py`)

Why the labels changed. Under the original rule (airborne = all four foot joints above fixed
heights), the 50-step model's seed 0 had 418 airborne stretches of at least 0.2 s: 209 ballistic
(g > 5 m/s², 163 from jump prompts) and 195 near zero (|g| < 2, 191 from non-jump prompts):

| What the near-zero stretch was | Stretches | Typical prompts |
| --- | --- | --- |
| Another body part supports it (wrist or knee within 15 cm of the floor) | 48 | crawling, cartwheels |
| Feet raised but nearly still | 48 | balance beam, standing poses |
| Whole motion offset above the floor | 5 | |
| Unexplained: unsupported, not falling | 94 | walking in place, walking backward, stumbling, balance beam (34) |

- The unexplained cases (`results/phase2/hovering/unexplained_examples.png`) are ordinary
  walking poses with the whole body 7 to 12 cm off the floor (median lowest foot 8.3 cm,
  median duration 0.6 s), while the same motions touch the floor elsewhere (median motion
  lowest foot 1.4 cm): a real floating artifact.
- A single foot-clearance threshold does not separate them: a peak lowest foot above 15 cm
  keeps 73% of ballistic stretches but also 44% of near-zero ones.

The fix (approved November 16):
- Airborne also requires no knee, wrist or pelvis within 15 cm of the floor
  (`SUPPORT_JOINTS`, `SUPPORT_HEIGHT_MIN = 0.15`).
- The per-sample gravity failure is split: **gravity error (jumps)** = mean |g − 9.81| over
  unsupported stretches that fall at 2 m/s² or more; **floating rate** = share of frames in
  unsupported stretches that fall slower than 2 m/s² (`FLOAT_G_MAX = 2.0`).
- Unchanged: in-flight (no foot in contact), float rate (contact frames with the lowest foot
  above 5 cm), and all other labels.
- Raised feet with no other support (balance beam, standing on something unseen) still count
  as airborne; they are a known remaining limitation of a scene-free label.

## 7. Early warning: abort and resample (`scripts/early_warning_sim.py`, November 16)

Simulated offline on the cached phase 2 samples (50-step: 3,000; 1000-step: 2,000). Each saved
sample already holds its decoded guess at every recorded step and its final motion; samples of
a prompt are pooled across seeds and redrawn as retries. Failure = the final motion has any
floating. Cost = denoising steps per delivered motion (1 = one full sample). Folders:
`results/phase2/early_warning/` (summary.csv, operating_points.csv, matched_failure_compute.csv,
tradeoff.png).

**The detector works without training.** Rule: abort if the decoded guess shows any floating.

| Check point | Flags | Precision (flagged that do float) | Recall (floaters caught) |
| --- | --- | --- | --- |
| 50-step, 20% (step 10) | 7.0% | 80% | 59% |
| 50-step, 40% (step 20) | 7.0% | 88% | 65% |
| 1000-step, 20% (step 200) | 9.3% | 83% | 45% |
| 1000-step, 40% (step 400) | 10.2% | 85% | 50% |

Phase 4's go criterion from RESEARCH_PLAN (beats text only; within-prompt ρ above 0.2 by 40%)
is met by the decoded guess: within-prompt ρ for floating rate 0.52 (50-step) and 0.37
(1000-step) at 20%, against text-only ρ of 0.25.

**The compute payoff is small.**
- Baselines: generating once delivers 9.4% (50-step) and 17.3% (1000-step) floating motions.
  Checking the final motion and regenerating ("finish, then filter", up to 10 tries) delivers
  3.0% at 1.40x compute and 3.8% at 1.62x.
- Early abort alone (no final check) bottoms out at about 6.7% and 14.5% floating for the
  guess rule. It is beaten by finish-then-filter at similar compute (2 tries: 5.4% at 1.09x;
  9.6% at 1.17x). The final check costs almost nothing, so it should always be kept.
- Early abort plus the final check, against finish-then-filter at the same delivered failure:

| Model | Delivered floating | Saving, guess rule | Saving, layer probe (same abort share) | Saving, perfect detector |
| --- | --- | --- | --- | --- |
| 50-step | 5.4% | 3.0% | 3.2% | 6.7% |
| 50-step | 4.5% | 5.1% | 5.4% | 10.3% |
| 50-step | 3.6% | 4.6% | −0.2% | 13.8% |
| 1000-step | 9.6% | 3.7% | 3.7% | 11.6% |
| 1000-step | 6.9% | 5.8% | 6.3% | 16.8% |
| 1000-step | 5.1% | 8.6% | 10.5% | 22.8% |

- Random aborts at the same rates give no gain (control).
- 2.6% to 3.3% of delivered motions still float after 20 tries: some prompts (balance beam,
  walking in place) float in nearly every sample, so resampling cannot fix them.
- Interpretation: decoding the guess detects floating early with no training and matches an
  internal-layer probe, which is the comparison Tang et al. did not run. For MDM the saving is
  small (3 to 9% at matched failure; a perfect detector would save 7 to 23%) because generation
  is cheap, floating hits 9 to 17% of samples, and much of it is tied to the prompt.

## 8. Video replication: feasibility (November 16, no downloads)

Facts (Hugging Face and GitHub metadata, the IntPhys site, the Esmati et al. repository README
and `inference.py`):
- Model: Wan 2.1 T2V 1.3B, Apache-2.0, not gated. Minimum download about 17.6 GB (text encoder
  11.4 GB in bf16, video model 5.7 GB, VAE 0.51 GB); the Diffusers copy is 28.9 GB because its
  text encoder is stored in fp32 (about 23 GB, more than this Mac's 24 GB memory when loaded).
- Esmati et al. code: `github.com/ParsaEsmati/videodiffusionphysics`, MIT licence. IntPhys
  setting: 256 x 256, 100 frames, 100 inversion steps, guidance 1, features mean-pooled over
  tokens per block. The code picks CUDA or CPU (no Apple GPU path) and uses bf16 only on CUDA.
- Data: IntPhys dev set, CC BY 4.0, no registration, 3 GB (`dev.tar.gz`, all blocks),
  plausibility labels included.
- Mac speed: no PyTorch timing found for an M4 Pro. One MLX implementation reports about 110 s
  for a 480 x 832 x 81-frame generation of the 1.3B model; the PyTorch Apple GPU path is
  reportedly slower. A ComfyUI issue reports Wan 2.1 output silently corrupting on M4 Pro under
  MPS at some clip lengths.

Assessment:
- In Esmati et al.'s setup the "output" is the real video itself (they invert real clips), so
  "knows more than it shows" cannot be tested there the way we test MDM. What can be tested is
  their supporting claim that plausibility is "absent from the VAE latent input": that rests on
  a linear probe on the mean-pooled clean latent, which removes all temporal structure.
- Stage A (cheap): encode the IntPhys dev videos with the Wan VAE alone and read plausibility
  from the clean latent with (1) their linear probe on the fully pooled latent, (2) a linear
  probe that keeps the time axis, and (3) a small nonlinear temporal probe. Needs about 3.5 GB
  of downloads (VAE 0.51 GB, IntPhys dev 3 GB) and about an hour of encoding. The payoff is
  asymmetric: above-chance readout would directly undercut the "absent from the input" claim;
  chance readout leaves it standing (a small from-scratch probe on about 600 training videos is
  a weak test).
- Stage B (expensive): on-trajectory replication, generating videos and recording activations
  as we did for MDM. Needs the full 17.6 GB, a physics label measurable from generated video (no
  off-the-shelf one; would need object tracking, as in Kneifl et al.'s gravity setup), many hours
  of sampling, and the M4 Pro corruption risk. Not recommended now; revisit only on a rented GPU
  and only if Stage A is positive.

**Stage A results (November 16; `scripts/intphys_vae_encode.py`, `scripts/intphys_vae_probe.py`,
`results/intphys_vae/`).** Downloads approved: Wan 2.1 VAE (0.48 GB) and IntPhys dev (3.2 GB,
MD5 matches the published 6a72715007f2f3b0e9546bfa8d8fc39b). 360 videos, 90 scenarios, each
with 2 possible and 2 impossible versions; 97 frames at 256 x 256, latent [16, 25, 32, 32]. The
VAE on the Apple GPU matches CPU to 1.4e-6 relative error (worst latent frame and channel
5e-6), reconstruction PSNR 36 to 42 dB. Linear probe = ridge classifier, penalty chosen by
grouped inner CV. Splits keep a scenario's four videos together (grouped 5-fold, 10 repeats).

| Readout of the clean VAE latent (all blocks) | Per-video accuracy | Pairwise accuracy |
| --- | --- | --- |
| Pooled over time and space, linear (their baseline as read) | 50.1% ± 0.5 | 52.1% ± 1.5 |
| Time kept (spatial mean per frame), linear | 51.5% ± 0.7 | 50.4% ± 1.3 |
| Time + 4x4 space, linear | 50.9% ± 1.1 | 54.4% ± 1.2 |
| **Time + 4x4 space + frame-to-frame changes, linear** | **62.1% ± 0.9** | **79.6% ± 2.3** |
| Small nonlinear temporal probe (1D conv) | 50.4% ± 0.3 | 53.9% ± 2.5 |
| Label-shuffle null for the best linear readout (20 shuffles: mean, 95th percentile) | 49.7%, 51.7% | 49.0%, 53.3% |

- Per block, the change-feature probe reaches 79.2% (O1), 82.0% (O2) and 80.0% (O3)
  pairwise; per-video 58.8% to 63.5%. It beat all 20 label shuffles (p ≤ 0.048, the smallest
  this test can give with 20 shuffles).
- Their baseline as I read it is reproduced: the fully pooled latent is at chance.
- Not a global splice artifact: mean frame-to-frame change is 1.00 to 1.02 times higher in
  impossible videos at every step, and a training-free "largest change" score is at 50.0%
  pairwise. The probe uses localized changes, consistent with the impossible event itself
  (an object vanishing, appearing or changing in one region).
- The nonlinear probe did not train (per-video exactly 50% in two blocks: it predicts one
  class); with about 290 training videos that is a training failure, not evidence.
- Not comparable: Esmati et al. give IntPhys-only DiT accuracies in figures only; the text
  gives 81.27% per-video averaged over IntPhys and InfLevel for WAN-1.3B.
- Interpretation: the plausibility signal is not absent from the clean VAE latent. A linear
  readout that can see frame-to-frame change recovers 80% pairwise accuracy where full pooling
  gives chance, the same weak-readout pitfall as in MDM. This does not show the diffusion model
  adds nothing: its per-video readout may still be stronger than our 62%.

## 9. Open items

- The 1000-step model's extra floating rests on two seeds that agree; one more seed would
  firm it up (about 40 minutes of sampling).
- Phase 3 (causal steering) is demoted; revisit only if time remains.
- Re-check arXiv before writing; update HANDOFF.md and RESEARCH_PLAN.md.

## 10. Revised plan (proposed, November 16, 2026)

Working title: "Do motion diffusion models know more than they show? A controlled test."
Venue: TMLR (claims judged on support; negative results fit).

1. Done: fix the labels and rerun everything on cached arrays.
2. Make the fair-baseline protocol the method section: explicit label, linear and nonlinear
   readouts of the model's own guess, layers, guess + layer, within-prompt scoring, learning
   curves. Report it for every target on both checkpoints.
3. Done (section 7): early warning works with the decoded guess, matches a layer probe, and
   saves 3 to 9% of compute at matched failure. Report it as a practical side result.
4. Generality: Stage A done (section 8): plausibility is readable from Esmati et al.'s clean
   VAE latent (80% pairwise) once the readout keeps frame-to-frame change, so their "absent from
   the input" evidence shows the same weak-readout pitfall. Stage B not recommended now.
5. Demoted: causal steering (old phase 3). Optional, only if time remains.
6. Done: first full draft (paper/), second prose pass, arXiv recheck (November 16), SMPL Figure 1,
   label explainer figure, redesigned result figures.

**Next (agreed November 16).** Group 1, no downloads or cost, in progress:
- 1a. Third seed for the 1000-step model, to firm up "floats about twice as often".
- 1b. Confidence intervals and paired tests for the key comparisons (bootstrap over prompts).
- 1c. Threshold sensitivity: contact heights, the 15 cm support rule, the 2 m/s^2 floating cut-off.
- 1d. A temporal readout (whole-motion model) on both the estimate and the layers.
- 1e. A simulated TMLR reviewer pass on the draft by an independent agent.
- 1f. Release prep: reproduction README, code-release statement (the last TODO in the paper).

Group 2, needs the user: AMASS registration (real-motion calibration of the labels; downloads
need approval), and a TMLR account on OpenReview.

Group 3, bigger and optional: supplementary videos of SMPL renders (agreed November 16: do this
after the paper reaches its final version; about 2 to 3 hours plus 30 minutes of compute, MP4 needs
the 30 MB `imageio-ffmpeg` package); the on-trajectory video
replication (rented GPU, needs a cost estimate and approval); a second motion model.

Second-model candidates (checked November 16 from public repository metadata):
- DiP (MDM authors' autoregressive successor, in the MDM repository): MIT, checkpoint 232 MB on
  Hugging Face, DistilBERT text encoder, 10 diffusion steps, transformer decoder. Cheapest to add
  (same code base and labels). Catch: generation starts from a 2 s motion prefix sampled from
  HumanML3D data, which we only have for one clip; needs AMASS/HumanML3D or a generated prefix.
- MLD (motion latent diffusion, CVPR 2023): MIT. Diffusion in a learned motion latent, so its
  estimate lives in latent space as in video models; bridges to the video papers. Checkpoint size
  and Mac compatibility not yet checked.
- MoMask (masked token model, CVPR 2024): MIT. Not diffusion, but iterative unmasking gives a
  per-iteration estimate; tests generality beyond diffusion.
- Kimodo (NVIDIA, 2026): code Apache-2.0 (nv-tlabs/kimodo). Checkpoint licences and Apple-GPU
  support still to check (HANDOFF notes supported GPUs start at Ampere).

## 11. Revision after the simulated review (started November 16)

Plan agreed with the user: (a) frame the comparison as usable information, with the full
data-processing chain; (b) tuned readouts on both sides, pooled seeds, learning curves for the
fair readouts, the conditional-probing gain; (c) a jump-heavy sample for gravity; (d) a
within-prompt score that also removes timing; (e) correct the early-warning and IntPhys claims
and add the missing IntPhys checks; (f) missing related work.

**Label threshold sensitivity, physics (`scripts/label_sensitivity.py physics`,
`results/sensitivity/physics.csv`, `detector.csv`; rerun with all six runs).** Floating rate on
test captions, % of frames (seeds 0/1/2 for each model):

| Setting | 50-step | 1000-step |
| --- | --- | --- |
| default | 4.69 / 4.58 / 3.79 | 8.92 / 8.64 / 9.67 |
| contact heights x0.75 | 7.65 / 7.20 / 6.43 | 13.47 / 13.05 / 13.79 |
| contact heights x1.25 | 3.59 / 3.29 / 2.83 | 6.12 / 6.19 / 6.63 |
| support rule 10 cm | 4.75 / 4.60 / 3.82 | 9.15 / 8.70 / 9.67 |
| support rule 20 cm | 4.69 / 4.50 / 3.79 | 8.88 / 8.58 / 9.61 |
| floating cut-off 1 m/s^2 | 4.58 / 4.45 / 3.70 | 8.68 / 8.49 / 9.51 |
| floating cut-off 3 m/s^2 | 4.72 / 4.62 / 3.89 | 8.99 / 8.74 / 9.78 |

- The 1000-step model floats more than the 50-step model under every setting, in every seed:
  its lowest seed is above the 50-step model's highest (ratio of means 1.9 to 2.1). Contact
  heights move both rates the most; the support rule and the cut-off barely matter. Jump gravity
  on test captions stays 8.1 to 10.1 m/s^2 in every setting.
- Controls under each setting (`label_sensitivity.py controls`, seed 0 of each model, fixed MLP,
  20%; `results/sensitivity/<setting>/controls/`): dynamic balance (contact settings only) estimate
  MLP - layer linear +0.020 to +0.059, gain from adding the layer -0.018 to +0.004, standard
  comparison (layer linear - explicit) +0.031 to +0.102; flight gravity gain +0.040 to +0.111 (layers
  6 / 3; with layer 5, rerun in round 3: +0.023 to +0.102).
  Same conclusions as the default. The floating cut-off does not enter the probed labels, so those
  runs equal the default.
- Rule "estimate shows floating at 20%": precision 0.66 to 0.94 and recall 0.44 to 0.69 over
  all settings and runs (default: 0.73 to 0.89, 0.44 to 0.68). Paper: Appendix table
  `tab_sensitivity.tex`.

**IntPhys: where the change-feature gain comes from (`scripts/intphys_vae_probe.py
--extra-only`, `results/intphys_vae/summary_extra.txt`).** Linear probes, all blocks, 10 repeats:

| Readout of the clean VAE latent | Per-video | Pairwise |
| --- | --- | --- |
| Time + 4x4 space + *signed* frame changes (a linear function of the latent) | 52.8% ± 1.1 | 52.2% ± 2.2 |
| Absolute 4x4 frame changes only | 61.8% ± 0.6 | 77.9% ± 1.7 |
| Time kept (spatial means) + their absolute changes | 53.4% ± 1.2 | 57.4% ± 2.2 |
| (earlier) time + 4x4 space + absolute changes | 62.1% ± 0.9 | 79.6% ± 2.3 |

- Signed changes add nothing (52.2%, null 95th percentile 53.1%, p = 0.19): plausibility is not
  linearly decodable from the latent, pooled or time-resolved. It becomes readable through one
  fixed nonlinearity, the magnitude of local frame-to-frame change (77.9% from that alone,
  78.5 to 81.0% per block). Without the 4x4 grid the gain mostly goes (57.4%).
- Correct claim: Esmati et al.'s linear baseline result holds even with time kept, but their
  conclusion that the signal is constructed by the diffusion model does not follow: a fixed,
  training-free transform of the clean latent exposes it, and by data processing the clean latent
  bounds what any layer can carry. Recommendation 4 in the draft ("a time-resolved linear readout
  recovers") was wrong and is corrected.

**Recording detail confirmed for the data-processing chain.** Layers are recorded from the
conditional pass only (`tools/p2_recorder.py`); the delivered estimate is guided, so it is a
function of the conditional activations together with the noisy input (the unconditional pass
sees only x_t). Chain used in the paper: I(y; estimate) ≤ I(y; a_l, x_t) ≤ I(y; x_t, text).

**Jump-heavy prompts (`scripts/make_jump_prompts.py` → `prompts/jump_heavy.txt`).** The 14
curated flight prompts plus one caption (seed 0) from every HumanML3D test motion with a jump,
hop, leap or skip caption: 245 distinct prompts (231 test captions; 19 prompts also in the
phase 2 list). New options for the generated phase 2 script: `--prompts-file`,
`--samples-per-prompt`, `--frame-stride`, `--tag`, `--generate-only`; with none of them the
script is unchanged.

**Jump-heavy run (50-step, seeds 0 to 2, 245 prompts x 4 = 980 motions per seed, 493 s for
all three on the Apple GPU; `data/jump_heavy/`).** Per seed: 788 / 788 / 798 motions (80 to 81%)
have an airborne stretch of at least 0.2 s with a fitted gravity, against 171 to 192 (17 to 19%)
in the phase 2 runs; 4,557 / 4,484 / 4,729 gravity-labelled recorded frames (phase 2: 1,661 /
1,565 / 1,484); 228 to 229 prompts with at least one such motion (phase 2: 70 to 82). Median
over motions of the per-motion median g: 9.14 / 9.23 / 9.07 m/s^2 (these stretches are mostly
real jumps; in phase 2 the same statistic is 1.5 to 4.4 because floating dominates).

**1000-step model, third seed (seed 2).** Floating on test captions 9.7% (seeds 0, 1: 8.9,
8.6); on jump prompts 4.8% (9.3, 5.2); jump gravity on jump prompts 9.22 m/s^2 (9.18, 9.13).
"Floats about twice as often as the 50-step model" holds in all three seeds (50-step test
captions: 3.8 to 4.7%).

**Tuned readouts (`scripts/fair_readouts.py`).** Seeds pooled, folds grouped by prompt across
seeds; ridge (penalty by grouped validation), an MLP family with a linear skip (5
configurations, chosen by grouped validation), and the better of the two by validation; scores
pooled, within prompt, and within prompt and frame; prompt-bootstrap CIs (1,000 replicates).

50-step model, seeds 0 to 2 pooled, 20% of denoising (`results/phase2/fair/50steps_*.csv`).
Pooled score (R2; AUC for flight):

| Readout | Dynamic balance | Static balance | In flight | Flight gravity |
| --- | --- | --- | --- | --- |
| explicit | 0.784 | 0.771 | 0.876 | 0.708 |
| estimate, ridge / tuned | 0.866 / **0.889** | 0.825 / **0.830** | 0.974 / **0.976** | 0.788 / 0.799 |
| layer, ridge / tuned | 0.850 / 0.863 | 0.780 / 0.800 | 0.977 / 0.971 | 0.857 / **0.860** |
| estimate + layer, ridge / tuned | 0.880 / 0.884 | 0.835 / 0.825 | 0.980 / 0.972 | 0.864 / 0.859 |

Differences with 95% CIs, generated from `50steps_bootstrap.csv`:

| Comparison (pooled; within prompt; within prompt and frame) | Dynamic balance | Static balance | In flight | Flight gravity |
| --- | --- | --- | --- | --- |
| estimate tuned - layer tuned | +0.026 [0.016, 0.038]; +0.019 [0.015, 0.024]; +0.026 [0.020, 0.032] | +0.029 [0.020, 0.038]; +0.028 [0.020, 0.038]; +0.032 [0.023, 0.042] | +0.005 [-0.000, 0.011]; +0.023 [0.015, 0.033]; +0.024 [0.016, 0.035] | -0.061 [-0.113, -0.019]; -0.100 [-0.199, 0.003]; -0.128 [-0.248, 0.001] |
| gain, tuned | -0.005 [-0.010, -0.000]; -0.003 [-0.006, -0.001]; -0.004 [-0.007, -0.000] | -0.004 [-0.008, -0.000]; -0.005 [-0.010, -0.000]; -0.003 [-0.008, 0.002] | -0.004 [-0.009, 0.002]; -0.009 [-0.017, -0.002]; -0.010 [-0.019, -0.002] | +0.059 [0.029, 0.099]; +0.078 [0.012, 0.144]; +0.084 [0.009, 0.151] |
| gain, ridge only | +0.014 [0.010, 0.019]; +0.014 [0.009, 0.019]; +0.018 [0.013, 0.025] | +0.010 [0.005, 0.015]; +0.010 [0.006, 0.013]; +0.011 [0.006, 0.017] | +0.006 [0.003, 0.009]; +0.013 [0.008, 0.018]; +0.014 [0.008, 0.019] | +0.075 [0.048, 0.116]; +0.107 [0.042, 0.170]; +0.116 [0.032, 0.191] |
| standard: layer ridge - explicit | +0.067 [0.027, 0.109]; +0.062 [0.024, 0.108]; +0.075 [0.025, 0.129] | +0.010 [-0.022, 0.034]; +0.001 [-0.027, 0.026]; -0.017 [-0.047, 0.009] | +0.102 [0.074, 0.134]; -0.031 [-0.055, -0.010]; -0.038 [-0.062, -0.016] | +0.149 [0.049, 0.276]; +0.067 [-0.078, 0.241]; +0.020 [-0.180, 0.239] |

- Balance and flight: with seeds pooled and every readout tuned on held-out prompts, the
  estimate's readout beats the layer's, and adding the layer to it does not help (the tuned gain
  is slightly negative, so the tuned combined readout does not fully use its larger input).
  Within linear readouts the layer does add a little (+0.006 to +0.014), less than what the
  estimate gains from a nonlinear readout. Usable information here depends on the family.
- Flight gravity: the layer adds to the estimate, also within prompt and within prompt and frame
  (+0.084 [0.009, 0.151]). Caveat to test:
  all these readouts see one frame; gravity is a second time derivative, which one frame of the
  estimate cannot show but the layer can draw from other frames through attention. The
  temporal readout (whole motion, both sides) and the jump-heavy data test this.
- Standard comparison for flight flips sign within prompt (-0.031 [-0.055, -0.010]; within
  prompt and frame -0.038 [-0.062, -0.016]): the
  layer's AUC advantage over the explicit label comes from prompt-level information.
- Tuning picked the MLP family in 60 to 100% of folds; every configuration was chosen at least
  once (`50steps_chosen.csv`).

1000-step model, seeds 0 to 2 pooled, 20% of denoising (`results/phase2/fair/1000steps_*.csv`;
77,797 / 77,797 / 90,000 / 9,425 frames). Pooled score:

| Readout | Dynamic balance | Static balance | In flight | Flight gravity |
| --- | --- | --- | --- | --- |
| explicit | 0.655 | 0.548 | 0.788 | 0.692 |
| estimate, ridge / tuned | 0.746 / **0.779** | 0.641 / **0.671** | 0.919 / **0.935** | 0.691 / 0.669 |
| layer, ridge / tuned | 0.720 / 0.765 | 0.592 / 0.647 | 0.934 / 0.932 | 0.687 / **0.715** |
| estimate + layer, ridge / tuned | 0.764 / 0.778 | 0.653 / 0.665 | 0.940 / 0.938 | 0.703 / 0.712 |

| Comparison (pooled; within prompt; within prompt and frame) | Dynamic balance | Static balance | In flight | Flight gravity |
| --- | --- | --- | --- | --- |
| estimate tuned - layer tuned | +0.015 [0.008, 0.022]; +0.010 [0.006, 0.015]; +0.013 [0.008, 0.018] | +0.024 [0.014, 0.034]; +0.020 [0.013, 0.027]; +0.022 [0.014, 0.029] | +0.003 [-0.002, 0.008]; +0.018 [0.007, 0.032]; +0.020 [0.007, 0.034] | -0.047 [-0.086, -0.017]; -0.006 [-0.030, 0.022]; +0.003 [-0.030, 0.039] |
| gain, tuned | -0.002 [-0.006, 0.002]; -0.001 [-0.004, 0.001]; -0.002 [-0.005, 0.001] | -0.006 [-0.018, 0.002]; -0.004 [-0.012, 0.002]; -0.004 [-0.014, 0.003] | +0.003 [-0.003, 0.009]; -0.000 [-0.008, 0.007]; +0.001 [-0.008, 0.009] | +0.043 [0.021, 0.074]; +0.007 [-0.008, 0.023]; +0.006 [-0.015, 0.025] |
| gain, ridge only | +0.018 [0.014, 0.021]; +0.013 [0.010, 0.015]; +0.014 [0.010, 0.017] | +0.013 [0.007, 0.018]; +0.009 [0.005, 0.014]; +0.009 [0.004, 0.014] | +0.021 [0.015, 0.027]; +0.016 [0.011, 0.021]; +0.016 [0.010, 0.022] | +0.013 [-0.013, 0.036]; +0.001 [-0.035, 0.029]; +0.005 [-0.035, 0.038] |
| standard: layer ridge - explicit | +0.064 [0.037, 0.094]; +0.047 [0.027, 0.068]; +0.051 [0.029, 0.074] | +0.045 [0.007, 0.086]; +0.027 [-0.005, 0.064]; +0.022 [-0.008, 0.058] | +0.146 [0.118, 0.177]; +0.020 [-0.008, 0.050]; +0.022 [-0.010, 0.055] | -0.005 [-0.060, 0.060]; -0.054 [-0.110, -0.006]; -0.059 [-0.117, -0.011] |

- Balance and flight replicate the 50-step result.
- Gravity: the layer's advantage on the 1000-step model is prompt-level only (gain +0.043
  pooled, +0.006 [-0.015, 0.025] within prompt and frame); on the 50-step model it held within
  prompt and frame (+0.084 [0.009, 0.151]).
- Correction (November 16): an earlier version of these notes quoted the within-prompt column as
  "within prompt and frame"; the tables above are now generated from the CSVs.

Best of family (`scripts/fair_best_of_family.py`, `results/<tag>/fair/best_of_family.csv`): per
feature set, whichever of ridge / MLP / tuned scores best on the test frames, CIs with that choice
fixed. Gain from adding the layer, pooled: balance and flight -0.006 to +0.005 on both models,
every interval covering zero or below it; gravity +0.045 [0.019, 0.084] (50-step) and +0.029
[-0.014, 0.074] (1000-step); within prompt and frame the 50-step gravity gain is +0.066
[-0.003, 0.134].

Jump-heavy data, 50-step, seeds pooled (`results/jump_heavy/fair/`; gravity 13,770 frames, 236
prompts; flight 88,200 frames):

| Readout | Flight gravity (R2; within prompt and frame) | In flight (AUC; within prompt and frame) |
| --- | --- | --- |
| explicit | 0.389; 0.458 | 0.926; 0.811 |
| estimate, tuned | 0.710; 0.684 | 0.984; 0.881 |
| layer, tuned | 0.716; 0.683 | 0.981; 0.874 |
| estimate + layer, tuned | 0.731; 0.708 | 0.982; 0.880 |

- Gravity: estimate and layer tie (-0.006 [-0.023, 0.011]; within prompt and frame +0.000
  [-0.022, 0.024]); adding the layer gains +0.021 [0.009, 0.034] (within prompt and frame +0.024
  [0.010, 0.040]); best of family the same. With 2.9 times the gravity frames, the layer's lead
  over the estimate in the main 50-step data (0.860 vs 0.799, about 84 training prompts with
  flight) is gone; a small complementary gain remains. Learning curve on these data: the layer
  leads at 25 and 50 training prompts (0.549 vs 0.405; 0.626 vs 0.571), the two meet by 100
  (0.676 vs 0.671) and at 188 (0.716 vs 0.710).
- The standard comparison would report a large layer advantage (layer ridge - explicit +0.301
  [0.233, 0.380]) because the explicit gravity of the 20% estimate is poor (R2 0.389).
- Flight replicates: estimate ahead (+0.002 [0.001, 0.004]; within +0.007), no gain from adding
  the layer.

**Tuned learning curves (`results/<tag>/fair/<checkpoint>_curve.csv`, 2 draws per size).**
Dynamic balance, tuned estimate vs tuned layer: 50-step 0.757 / 0.617 at 25 prompts (explicit
0.783 wins), 0.854 / 0.773 at 50, 0.874 / 0.814 at 100, 0.889 / 0.863 at 200; 1000-step 0.684 /
0.578, 0.728 / 0.695, 0.766 / 0.740, 0.779 / 0.765. The estimate's lead narrows with data (50-step
0.060 at 100 to 0.026 at 200; 1000-step 0.026 to 0.014), so the layer could catch up with much
more data; the gain from adding the layer stays at or below zero at every size (-0.020 to
0.000). Stated in the paper (Section 4.4).

**Temporal readout on the jump-heavy gravity (`scripts/temporal_readout.py --tag jump_heavy
--targets flight_gravity`; `results/jump_heavy/controls/50steps_temporal_readout.csv`; per seed,
not pooled).** R2 estimate / layer / both: seed 0 0.686 / 0.687 / 0.669; seed 1 0.657 / 0.684 /
0.670; seed 2 0.656 / 0.711 / 0.705; mean 0.666 / 0.694 / 0.682 (sd 0.017 / 0.015 / 0.020). Within
prompt: 0.660 / 0.682 / 0.677. A readout that sees the estimate's whole motion does not close the
gravity gap; the layer leads in every seed (by 0.001, 0.027, 0.055). Verdict for gravity: a small
real layer advantage, 0.02 to 0.03 R2, in usable information beyond our readouts of the estimate.

**Temporal readout, main data (`scripts/temporal_readout.py`, `results/phase2/controls/
<checkpoint>_temporal_readout.csv`; per seed, 20%).** Mean R2 (AUC for flight) estimate / layer
/ both: 50-step dynamic balance 0.859 / 0.833 / 0.849, flight 0.964 / 0.960 / 0.963, gravity
0.768 / 0.822 / 0.787; 1000-step 0.737 / 0.720 / 0.735, 0.915 / 0.910 / 0.921, 0.648 / 0.695 /
0.681. Estimate minus layer per seed: dynamic balance +0.014 to +0.034 (all six runs), flight
+0.001 to +0.010 (all six), gravity -0.081 to +0.004 (layer ahead in five of six). A readout that
sees the whole motion gives the same ordering as the per-frame readouts.

**Per-seed prompt bootstrap of the fixed-MLP comparison (`scripts/bootstrap_ci.py`,
`results/phase2/controls/bootstrap_ci.csv`; 2,000 replicates, seed-averaged).** Estimate MLP -
layer linear, dynamic balance: +0.040 [0.031, 0.050] (50-step), +0.050 [0.042, 0.059]
(1000-step); static +0.057 / +0.063, intervals above zero. Gain from adding the layer: balance
-0.025 to -0.001 (intervals below zero or covering it), flight -0.004 [-0.013, 0.002] and +0.008 [0.002,
0.014], gravity +0.112 [0.063, 0.186] and +0.066 [0.044, 0.092]. Agrees with the pooled tuned
results; the tuned, pooled CIs are the ones reported in the paper.

**Early warning rerun with all 6,000 samples (`scripts/early_warning_sim.py`, 1000-step now
three seeds; `results/phase2/early_warning/`).** At 20% of denoising: rule precision 80% (50-step)
and 82% (1000-step), recall 59% and 47%; at 40% precision 88% and 84%. Generate once: 9.4% and
17.7% float. Compute saved at matched failure (finish-then-filter budgets 2 / 3 / 5):

| | 50-step | 1000-step |
| --- | --- | --- |
| estimate rule | 3.0 / 5.1 / 4.6% | 4.2 / 6.9 / 9.7% |
| layer probe (layer 3; layer 6) | 3.2 / 5.4 / -0.2% | 4.1 / 7.1 / 10.9% |
| perfect detector | 6.7 / 10.3 / 13.8% | 12.0 / 17.1 / 22.5% |

- Corrections to the draft: the layer probe is about equal to the rule (0.2 to 1.2 points more in
  four of six comparisons, 0.02 less in one, and fails at the strictest 50-step target), not
  "no better". Early abort without the final check is not dominated at 20%: within 0.9 points of
  finish-then-filter at equal compute (mixing budgets 1 and 2); it falls behind at 40%. Savings
  grow with stricter targets on the 1000-step model. After 20 tries 2.6 to 3.6% still float.
  Random aborts leave failure at the base rate (9.2 to 9.6%, 17.5 to 18.0%).
- Within-prompt Spearman for the floating rate at 20% (mean over seeds): estimate 0.52 and 0.40,
  fixed layer 0.28 and 0.30; prompt alone across prompts 0.25 and 0.23.

**Second simulated review: major revision, confidence 4/5.**
Verified against the result files before acting:
- The gravity target is mostly floating: 76% (50-step) and 83% (1000-step) of gravity-labelled
  frames in the main data are floating stretches (g < 2), 15% in the jump-heavy data. Its R2
  mostly measures floating versus falling. On falling frames only (g >= 2), from the saved tuned
  predictions: main 50-step explicit -1.97, estimate -0.32, layer 0.16, both 0.12; 1000-step all
  negative; jump-heavy explicit -0.14, estimate 0.28, layer 0.34, both 0.34. "Gravity during
  jumps R2 = 0.82" in the contributions is therefore misleading.
- The fixed gravity layers (6 for 50-step, 3 for 1000-step) were chosen under the original labels
  (v1 best: layer 6 0.795; layer 3 0.674). Under the current labels the best mean linear layer at
  20% is layer 5 (50-step 0.814 vs 0.802 for layer 6; 1000-step 0.686 vs 0.673 for layer 3).
- Paper error: the flight lead "within prompt and frame" was quoted as 0.023 and 0.018 (those are
  within prompt); within prompt and frame it is 0.024 and 0.020.
- Not yet checked or addressed: conditional gain for every layer and all layers together;
  raw-feature comparison; estimator of the gain that can fall back to the estimate alone; the
  estimate readout in the early-warning matched comparison; scripts for the IntPhys global-change
  numbers (1.00 to 1.02, "largest change" score), which are not in scripts/ or results/.

## 12. Revision round 3 (after the second simulated review, started November 16)

**Early warning with common random numbers (`scripts/early_warning_sim.py`; every strategy now
uses the same random draws).** The earlier runs drew all strategies from one random stream, so
adding a score shifted every later number; savings moved by up to 0.4 points between runs, and the
layer probe's -0.2% at the strictest 50-step target was simulation noise. Corrected, at 20%,
matched to finish-then-filter budgets 2 / 3 / 5 (50-step 5.5 / 4.5 / 3.8% floating; 1000-step
9.4 / 6.8 / 4.6%):

| Score | 50-step | 1000-step |
| --- | --- | --- |
| estimate rule (training-free) | 3.4 / 4.8 / 6.7% | 4.3 / 6.9 / 9.9% |
| estimate readout (ridge, frame-averaged, now reported) | 3.6 / 5.1 / 6.8% | 4.4 / 6.9 / 9.8% |
| layer probe (layer 3; layer 6) | 3.6 / 5.1 / 4.5% | 4.0 / 6.8 / 10.7% |
| readout + layer (conditional, new) | 3.8 / 5.5 / 4.5% | 3.7 / 6.5 / 10.2% |
| perfect detector | 6.7 / 10.2 / 14.9% | 11.9 / 17.0 / 22.5% |

- Early-abort floors without the final check: rule 6.6 to 6.9% and 13.6 to 14.5%; trained readouts
  5.5 to 6.3% and 10.3 to 12.0%. Rule at 20% vs finish-then-filter at equal compute: -0.15 to
  +0.64 and -0.08 to +0.86 points; at 40% behind by 0.9 to 2.2 and 0.5 to 3.8 points.
- Ranking (`ranking.csv`), Spearman with the final floating rate after removing prompt means, 20%:
  rule 0.57 / 0.46, estimate readout 0.34 / 0.33, layer probe 0.26 / 0.28, readout + layer 0.29 /
  0.28. The layer adds nothing to the estimate's readout at the sample level either.
- 20 tries: 2.6 to 3.7% still float. Generate once 9.4% and 17.7%.

**IntPhys.** `scripts/intphys_change_stats.py` (`results/intphys_vae/change_stats.txt`): mean
|frame change| impossible / possible 1.000 to 1.033 per step (paper said 1.00 to 1.02; corrected);
largest global change and largest 4x4-cell change both 50.0% pairwise (training-free). 1,000-shuffle
null (`--null-only 1000`, `null_1000.txt`): time + 4x4 + |change| and |change| only beat all 1,000
(p = 0.001; null pairwise mean 49.9%, 95th pct 56.7%, max 61.7%); signed changes p = 0.31 pairwise,
0.063 per video.

**Gravity split (`fair_readouts.py`, new targets at the existing 2 m/s^2 cut-off; gravity layer now
5, the best under the current labels).** 50-step main data, seeds pooled:
- float_vs_fall (4,710 frames, 106 prompts): AUC explicit 0.931, estimate tuned 0.980, layer tuned
  0.988; gain with fallback +0.008 [0.001, 0.015], within prompt and frame +0.050 [-0.007, 0.099].
- fall_gravity (falling stretches only; 1,147 frames, 57 prompts): explicit 0.171, estimate tuned
  -0.168, layer tuned 0.217, estimate + layer ridge 0.223; fallback gain +0.253 [0.060, 0.505],
  within prompt and frame +0.046 [-0.041, 0.127]. Too little data: the tuned estimate readout
  overfits below the one-feature explicit label.
- Balance and flight with the fallback estimator: gain +0.000 [-0.001, 0.001] (dynamic), -0.001
  (static), -0.004 (flight, AUC); raw estimate features tuned tie the tuned layer (dynamic +0.007
  [-0.004, 0.018], static -0.002, flight +0.004); the labels computed from the estimate add +0.019
  (dynamic). Refits (validation split and initialization varied), dynamic balance: fallback gain
  +0.000 / -0.000 / +0.003; estimate - layer +0.026 / +0.016 / +0.033.

1000-step main data, same analysis (`results/phase2/fair/1000steps_*`): fallback gain dynamic
balance -0.002 [-0.005, -0.000], static +0.000, flight -0.000 (AUC); estimate - layer +0.015
[0.008, 0.022], +0.024 [0.014, 0.034], flight +0.003 [-0.002, 0.008]; raw estimate features tie
the layer (-0.003, -0.002, +0.000). float_vs_fall: AUC estimate 0.964, layer 0.976, gain +0.003
[-0.001, 0.008]. fall_gravity: 1,620 frames, 78 prompts, estimate -0.069, layer 0.090, gain +0.082
[-0.035, 0.167] (within prompt and frame +0.106 [0.020, 0.195]); too little data, as on 50-step.

**Jump-heavy layer sweep (50-step; `results/jump_heavy/fair/50steps_sweep_*`).** fall_gravity
(11,640 frames, 233 prompts): estimate tuned R2 0.356; estimate + layer with fallback 0.356 (layers
0, 1), 0.370 (2), 0.419 (3), 0.432 (4), 0.435 (5), 0.436 (6), 0.438 (7), 0.439 (all layers). Gain
+0.064 to +0.083 for layers 3 to 7 and all, intervals above zero, and within prompt and frame
+0.056 to +0.084. float_vs_fall: no gain at any layer (-0.007 to +0.002, intervals covering zero).
**This is the one genuine case**: for the fall rate of real jumps, layers 3 to 7 carry usable
information that our per-frame readouts of the estimate (which include its own fitted g) do not
extract. Temporal readout on these targets queued.

**50-step main-data layer sweep (`results/phase2/fair/50steps_sweep_*`; flight finished after the
interruption into `*_sweep_part2_*`).** Gain with fallback, pooled: dynamic balance 0.000 at every
layer 0 to 7 and for all layers together (intervals within -0.005 to +0.004); floats or falls
-0.002 to +0.010; fall rate (1,147 frames, too few) 0.06 to 0.38 with wide intervals, not used.

**Jump-heavy 50-step, default spec (gravity layer 5; `results/jump_heavy/fair/50steps_*`).**
fall_gravity (11,640 frames): explicit 0.152, estimate raw tuned 0.267, estimate tuned 0.351,
layer ridge 0.374, layer tuned 0.427, estimate + layer ridge 0.415, with fallback 0.428. Gain with
fallback +0.077 [0.051, 0.107], within prompt and frame +0.073 [0.045, 0.098]; refits +0.108 and
+0.081. Estimate - layer -0.075 [-0.107, -0.043] (refits -0.089, -0.072). Curve (training prompts
25 / 50 / 100 / 186): estimate tuned -0.204 / 0.190 / 0.301 / 0.351, layer tuned 0.169 / 0.304 /
0.377 / 0.427: the gap is 0.076 and 0.075 at 100 and 186 prompts (0.113 at 50), so unlike the mixed target it is not
closing with data. Temporal readout (per seed, `results/jump_heavy/controls/50steps_temporal_readout_split.csv`):
fall rate estimate 0.278 / 0.304 / 0.311 (mean 0.297, sd 0.017), layer 0.363 / 0.365 / 0.408 (0.379,
0.025), both 0.338 / 0.392 / 0.375 (0.368); layer ahead by 0.062 to 0.097 in every seed; within prompt
0.442 / 0.513 / 0.511. Floats or falls: estimate 0.967, layer 0.965, both 0.967 (AUC). float_vs_fall: AUC estimate 0.974, layer 0.969, gain -0.000; in_flight gain
-0.001. Standard comparison for the fall rate: layer ridge - explicit +0.222 [0.180, 0.262].
**Verdict: the fall rate of real jumps is the one property where MDM's layers hold usable
information beyond every readout of its estimate we tried (+0.08 R2), robust to tuning, fallback,
refits, within prompt and frame, layer choice (3 to 7), a temporal readout and data size; balance,
flight and floating show none.**

**Gravity reruns with layer 5 (consistency).** Per-seed controls (`phase2_controls.py --target
flight_gravity --layer 5`) and the per-seed bootstrap rerun; temporal readout of the mixed flight
gravity on the main data (`*_temporal_readout_gravity5.csv`): layer minus estimate +0.054 / +0.057 /
+0.079 (50-step) and +0.011 / +0.076 / +0.073 (1000-step), the layer ahead in all six runs (with
layers 6 and 3 it was five of six).

**1000-step jump-heavy run (seeds 0 to 2, 980 motions each; 8,085 s on the shared Apple GPU).**
84 to 86% of motions have a fitted airborne stretch; 6,411 / 6,615 / 6,494 gravity-labelled
frames, of which 4,502 / 4,400 / 4,542 fall (13,444 pooled); 30 to 34% of the airborne frames float
(50-step jump-heavy: 15%).

**1000-step jump-heavy readouts (`fair_readouts.py --checkpoint 1000steps --tag jump_heavy`, layer
5; `results/jump_heavy/fair/1000steps_*`).** fall_gravity (13,444 frames, 240 prompts; 69% of the
19,520 gravity-labelled frames fall): explicit 0.161, estimate raw tuned 0.195, estimate tuned
0.287, layer ridge 0.273, layer tuned 0.309, estimate + layer tuned 0.370, with fallback 0.363. Gain
with fallback +0.076 [0.053, 0.096], within prompt +0.051 [0.032, 0.072], within prompt and frame
+0.021 [0.006, 0.037]. Estimate - layer -0.022 [-0.045, 0.004] pooled, but +0.041 [0.017, 0.063]
within prompt and frame: unlike the 50-step model, the layer alone does not beat the estimate; the
two carry partly different information. Curve (training prompts 25 / 50 / 100 / 192; mean of two
draws, one at 192): layer - estimate 0.185 / 0.026 / -0.000 / 0.022; gain with fallback 0.170 /
0.022 / 0.023 / 0.076 (estimate tuned -0.120 at 25 prompts). Standard comparison +0.113 [0.068,
0.165]. float_vs_fall: AUC estimate 0.966, gain with fallback +0.001 [-0.001, 0.003]; in_flight
+0.001; mixed flight_gravity +0.013 [0.005, 0.023].
Layer sweep (`1000steps_sweep_*`), fall rate gain with fallback: layers 0 / 1 / 2 -0.000 / +0.012 /
+0.047, layers 3 to 7 +0.066 / +0.060 / +0.076 / +0.080 / +0.071, all +0.076; within prompt and
frame +0.010 to +0.029 for layers 3 to 7 (interval above zero for 5, 6, 7). float_vs_fall: at most
+0.007 (all layers, [0.003, 0.011]).
Temporal readout (`results/jump_heavy/controls/1000steps_temporal_readout_split.csv`), fall rate per
seed: estimate 0.393 / 0.167 / 0.312 (mean 0.290, sd 0.114), layer 0.440 / 0.204 / 0.388 (0.344,
0.124), both 0.428 / 0.195 / 0.365; layer ahead by 0.047 / 0.038 / 0.076.
**Verdict: the pooled gain replicates on the 1000-step model (+0.08 R2, layers 3 to 7; temporal
readout layer ahead in every seed), but it is smaller within prompt and frame (+0.02) and with fewer
training prompts (+0.02 at 50 and 100).**

**Floating within prompt (50-step main data, 106 prompts with airborne stretches).** The layer adds
+0.054 [0.013, 0.089] to the within-prompt correlation of float_vs_fall (+0.050 [-0.007, 0.099]
within prompt and frame); on the jump prompts (236 prompts) +0.005 [-0.007, 0.019]. The abstract and
intro now claim within-prompt-and-frame parity only for balance and flight.

**Layer sweeps on the main data.** 50-step in_flight (run separately into `50steps_sweep_part2_*`
and appended to `50steps_sweep_*`): gain with fallback at most +0.003 (interval high 0.016).
1000-step (`results/phase2/fair/1000steps_sweep_*`): xcom at most +0.002, in_flight at most +0.007
[0.001, 0.014] (layer 7), float_vs_fall at most +0.011 [0.003, 0.019] (layer 5); fall_gravity
-0.011 to +0.213 with wide intervals (1,620 frames; too few, not used). Over all sweeps, no layer
adds more than 0.007 for dynamic balance and flight or 0.011 (AUC) for floats or falls.

**Condition token (`fair_readouts.py --tag cond_token --layer-spec cond`; the 50-step main runs
recorded again with `--record-cond-token`, identical motions and estimate scores).** Adding the
token's output at the same layer (broadcast to every frame) to estimate + layer, both tuned with
fallback: xcom -0.001 [-0.003, 0.002], in_flight +0.004 [-0.000, 0.009], float_vs_fall +0.002
[-0.003, 0.007]; with all eight layers 0.000 / 0.000 / -0.007. Fallback folds over the four data
sets: estimate alone in 25 of 40 for balance and flight, 1 of 10 for the jump fall rate.

**Labels on real motion capture (`scripts/amass_label_check.py`, `results/amass/`; AMASS BMLmovi,
downloaded by the user with the Extended SMPL+H model).** HumanML3D's preprocessing reproduced
(SMPL+H FK, 10 shape coefficients, 20 fps, y-up, uniform skeleton from a HumanML3D processed clip,
floor at the lowest joint); retargeted bone lengths match HumanML3D exactly. 1,801 clips, 86
subjects, 203,713 frames:
- Floating 0.09% of frames, 4 clips of 1,801 (MDM: 3.8 to 4.7% and 8.6 to 9.7% on test
  captions); 0.07 to 0.27% under every threshold setting. Raised contact 0.93% of contact frames
  (MDM 2.9 to 3.3% and 4.9 to 5.4%). The floating label has a false-alarm rate of about 0.1%.
- 302 airborne stretches >= 0.2 s in 129 clips, 5 of them floating (1.7%). Falling stretches
  (297): fitted g median 10.33, SD 1.06 m/s^2 in HumanML3D's processed form; at the subject's own
  scale (divided by HumanML3D's leg-length retargeting factor, median 1.067, range 0.92 to 1.23)
  median 9.65, SD 0.77. So HumanML3D's preprocessing makes real jumps fall about 7% fast on average,
  and MDM's generated jumps (9.1 to 9.8 on jump prompts) fall slower than its training data.
- Noise ceiling for the fall-rate R2 (1 - label noise variance / target variance, noise SD 0.685
  frame-weighted at subject scale, 0.984 retargeted): 0.91 / 0.81 on the jump-heavy data (target SD
  2.24), 0.93 / 0.85 on the main 50-step data. The best readouts reach 0.36 to 0.44, so the fall
  rate is hard to predict at 20% for a reason other than label noise.

**Checkpoints.** The two official checkpoints were trained with different releases of the MDM code;
their `args.json` files differ in many settings beyond the step count, so the paper treats them as
two models.

**Pimentel et al. (2020), checked against the paper (arXiv HTML v2):** Theorem 1 (Section 4.1)
uses the data-processing inequality to show BERT's embeddings carry no more information about a
property than the sentence; they measure gain over a control function (Section 3.2) and say ease
of extraction still needs a formal definition. The related-work sentence now says this.

## 13. Pre-submission experiments (November 17)

**Text control for the fall rate (`fair_readouts.py --layer-spec text`, jump-heavy runs, layer 5,
`results/jump_heavy/fair/<ckpt>_text_*`).** Feature sets: estimate; estimate + the prompt's CLIP
text embedding (`txt`, 512-d, broadcast to every frame: the model's own text input up to a linear
map); estimate + layer; estimate + text + layer. The fallback of the last also chooses among the
smaller sets (nested), so "layer adds beyond text" compares like with like.
- Text adds nothing usable: the tuned estimate + text readout scores below the estimate alone on
  held-out prompts (fall rate 0.233 vs 0.351 on 50-step, 0.179 vs 0.287 on 1000-step; ridge 0.232 vs
  0.273, 0.197 vs 0.206), and validation picks the estimate alone in 5 of 5 folds on both models,
  so the gain from text with fallback is 0.000.
- Layer adds beyond text, with fallback: 50-step +0.077 [0.052, 0.109], within prompt +0.076
  [0.054, 0.101], within prompt and frame +0.073 [0.044, 0.100]; 1000-step +0.076 [0.053, 0.096],
  +0.051 [0.032, 0.072], +0.021 [0.006, 0.037]: identical to the gain without text (the nested
  fallback picks estimate + layer in 5 and 4 of 5 folds).
- Floats or falls: no gain from text or layer (within -0.002 to +0.005).
- Caveat: the text readout learns from about 190 training prompts per fold, so it may use less of
  the text than the model does; the within-prompt score (Proposition 2 in `docs/math_notes.md`) is
  the stronger test, and it holds on the 50-step model.
**Verdict: the fall-rate gain is not the text prompt, as far as readouts trained on held-out prompts
can tell.**

**Step sweep for the fall rate (`fair_readouts.py --step K --no-raw`, layer 5 fixed, jump-heavy,
`results/jump_heavy/fair/<ckpt>_stepK_*`).** Tuned R2 (estimate / layer / both with fallback) and the
gain with fallback, pooled and within prompt and frame:
- 50-step, 20% (step 10): 0.351 / 0.427 / 0.428; +0.077 [0.051, 0.107]; +0.073 [0.045, 0.098]
- 40% (step 20): 0.474 / 0.457 / 0.517; +0.043 [0.027, 0.059]; +0.033 [0.018, 0.048]
- 60% (step 29): 0.580 / 0.485 / 0.598; +0.018 [0.009, 0.028]; +0.009 [0.003, 0.015]
- 80% (step 39): 0.743 / 0.521 / 0.748; +0.005 [-0.001, 0.013]; +0.002 [-0.001, 0.005]
- 1000-step, 20% (step 200): 0.287 / 0.309 / 0.363; +0.076 [0.053, 0.096]; +0.021 [0.006, 0.037]
- 40% (step 400): 0.378 / 0.342 / 0.421; +0.043 [0.022, 0.058]; +0.014 [-0.006, 0.031]
- 60% (step 599): 0.504 / 0.375 / 0.512; +0.008 [-0.003, 0.019]; -0.004 [-0.012, 0.004]
- 80% (step 799): 0.632 / 0.391 / 0.638; +0.006 [-0.003, 0.014]; -0.002 [-0.009, 0.005]
Explicit label 0.152 / 0.318 / 0.469 / 0.708 (50-step) and 0.161 / 0.220 / 0.357 / 0.460 (1000-step).
The estimate improves much faster than the layer, and the layer's extra information halves by 40%
and is gone by 60 to 80%: **the layers hold the fall rate about a fifth to two fifths of the
trajectory before the estimate shows it.**
Layer sweeps at 40% and 60% (`<ckpt>_sweep_stepK_*`, GPU; layer 5 was chosen at 20%), gain with
fallback, pooled: 50-step 40%: layers 3 to 7 +0.039 to +0.065 (best layer 4), all +0.057; 60%: at
most +0.019 (layer 6), all -0.001. 1000-step 40%: layers 2 to 6 +0.037 to +0.059 (best layer 5), all
+0.044; 60%: at most +0.011 (layer 6), all +0.002. Within prompt and frame at 40%: at most +0.043
(50-step) and +0.028 (1000-step). The same layer's gain differs between the CPU step run and the GPU
sweep by up to 0.014 (layer 5 at 50-step 40%: +0.043 vs +0.057), within the refit spread seen
before. **No layer adds more than 0.02 from 60% on.**

**MotionDiffuse, main data (`scripts/run_motiondiffuse.py`, 3 seeds x 1,000 motions, epsilon prediction,
DDPM 1000 steps, 20% = step 200; layers by `scripts/pick_layer.py`: xcom 3, balance 3, flight and gravity 6;
`results/phase2/fair/motiondiffuse_*`).** Tuned readouts, seeds pooled (score; gain with fallback):
- dynamic balance: estimate 0.823, layer 0.700 (estimate - layer +0.123 [0.107, 0.141]), gain 0.000 (the
  fallback picks the estimate in every fold); explicit label 0.124; standard comparison +0.496 [0.450, 0.541].
- static balance: estimate 0.729, layer 0.547 (+0.182 [0.157, 0.207]), gain 0.000.
- in flight (AUC): estimate 0.943, layer 0.905 (+0.038 [-0.003, 0.077]), gain -0.004 [-0.009, -0.001]
  (ridge scores higher AUCs than the tuned MLP here: 0.967 and 0.951; selection is by MSE).
- floats or falls (AUC; only 2,107 frames from 50 prompts): estimate 0.940, layer 0.988, gain +0.026 [0.007,
  0.055]; too few prompts for the within-prompt scores. Jump-heavy runs queued.
**Verdict so far: the main result replicates on an epsilon-predicting model with a different architecture,
more strongly (the estimate's readout beats the layer by 0.12 to 0.18 for balance).**

**MotionDiffuse, main-data layer sweep (`results/phase2/fair/motiondiffuse_sweep_*`, 20%, gain with fallback,
pooled).** Dynamic balance: no layer adds more than 0.001 (the fallback keeps the estimate). In flight: every
layer -0.010 to 0.000. Floats or falls: +0.016 to +0.034 per layer, all eight +0.044 [0.013, 0.087], but on only
50 prompts; the jump prompts below show none.

**MotionDiffuse, jump prompts (`data/jump_heavy/motiondiffuse_seed{0,1,2}`, 2,940 motions; tuned readouts, seeds
pooled; `results/jump_heavy/fair/motiondiffuse*`).** At 20% (step 200), layer 6:
- fall rate (7,431 falling frames, 235 prompts): explicit 0.289, estimate 0.365, layer 0.326, estimate + layer
  with fallback 0.390. Gain +0.025 [0.003, 0.047]; within prompt +0.022 [0.007, 0.038]; within prompt and frame
  +0.017 [-0.001, 0.036]. Estimate minus layer +0.039 [-0.003, 0.075]. MDM's gain is about 0.08 on both
  checkpoints, so MotionDiffuse's is about a third of it.
- floats or falls (AUC, 8,072 frames): estimate 0.827, layer 0.784, gain +0.001 [-0.017, 0.017].
- flight gravity: estimate 0.446, layer 0.433, gain +0.002 [-0.012, 0.016].
- in flight (AUC, 88,200 frames): estimate 0.975, layer 0.972, gain 0.000 [-0.001, 0.002].
- the fixed-configuration MLP diverged on the fall rate in fold 0 (validation MSE 2,324 for the estimate, 813
  for estimate + layer; pooled R2 -59.7 and -20.8); tuning chose ridge in that fold, so the tuned scores above
  are unaffected.
Layer sweep, fall rate (gain with fallback, pooled): layers 0 to 3 between -0.004 and 0.000, layer 4 +0.011,
5 +0.022 [0.004, 0.042], 6 +0.025, 7 +0.018 [-0.002, 0.038]; all eight +0.047 [0.021, 0.074]. Floats or falls: no
layer above +0.001 (layers 4 and 5 about -0.03).
Text control (fall rate): adding the text embedding to the estimate does not help (gain with fallback -0.028
[-0.049, -0.008]); the layer adds +0.053 [0.027, 0.081] beyond estimate + text, +0.024 [0.006, 0.045] within
prompt and frame. The text does not explain the gain.
Steps (fall rate, layer 6): 40% (step 400) estimate 0.630, layer 0.405, gain +0.012 [0.003, 0.022], within prompt
and frame +0.004 [-0.002, 0.009]; 60% (step 599) estimate 0.762, layer 0.417, gain +0.007 [-0.002, 0.016].
**Verdict: on MotionDiffuse the balance and flight results replicate, and the fall rate shows the same pattern
as MDM at a smaller size: a gain at 20% from the later layers that the text does not explain and that fades by
40 to 60% of denoising.**

## 14. Simulated review round (November 17, evening)

- **MotionDiffuse fall-rate refits** (`results/jump_heavy/fair/motiondiffuse_refit{1,2}_*`): gain with fallback
  refit 1 +0.022 [0.007, 0.041] (within prompt and frame +0.014 [0.005, 0.024]); refit 2 0.000, the fallback keeping
  the estimate in all five folds. With refit 0 (+0.025): 0 to 0.025 across three fits. Not robust.
- **Noise level** (`scripts/noise_levels.py`, `paper/tables/tab_snr.tex`): SNR at 20/40/60/80% is 0.104 / 0.517 /
  1.6 to 1.8 / 7.3 to 8.8 for MDM (cosine) and 0.002 / 0.027 / 0.240 / 1.910 for MotionDiffuse (linear). At matched
  noise (MotionDiffuse 60%, SNR 0.24) its fall-rate gain is +0.007 against MDM's +0.077 (20%) and +0.043 (40%).
- **Alignment** (50-step jump prompts): 92% of the 11,640 falling frames (7,125 falling stretches) have the 20%
  estimate airborne too; gain there +0.069 (R2 0.369 to 0.438); in the other 8% +0.140 (0.212 to 0.352).
- **Gain without fallback, fold counts, rescaled gains** (`paper/tables/tab_gain_detail.tex`): balance and flight
  without fallback within 0.003 of the estimate or below; the fallback keeps the estimate in 1 to 5 of 5 folds.
  Floats or falls without fallback: +0.011 (50-step), +0.011 [0.005, 0.019] (1000-step), +0.033 (MotionDiffuse).
  Rescaled: dynamic balance at most 0.001 nats; fall rate 0.063 / 0.056 nats (MDM jump prompts); floats or falls
  main data removes 39% / 10% / 43% of misranked pairs.
- **Video readouts tried**: ten (pooled, time kept, time + grid, + signed changes, + |change|, |change| alone, time
  kept + its |change|, temporal CNN, two training-free change scores); |change| is the only fixed nonlinearity tried;
  the |change| readouts are best in every block (78.5 to 82.0% pairwise, others at most 60.2%).
- **Positive control** (`fair_readouts.py --layer-spec inject`, 50-step; `results/{phase2,jump_heavy}/fair/50steps_inject_*`;
  `paper/tables/tab_control.tex`): the estimate readout's out-of-fold residual plus noise, worth 0.005 / 0.01 / 0.02 in
  R2 to its best linear use, added to the layer along a random direction. Dynamic balance: gains +0.013 [0.009, 0.017],
  +0.025, +0.040 (real layer +0.000 [-0.001, 0.001]): detected at 0.005. In flight: pooled AUC -0.010 / +0.001 / +0.001
  (not detected; AUC near ceiling, selection by MSE), within prompt and frame +0.018 / +0.042 / +0.072 (detected; real
  layer -0.008). Floats or falls, jump prompts: pooled +0.001 / +0.002 / +0.004 (not detected), within prompt and frame
  +0.006 [-0.014, 0.024] / +0.030 [0.002, 0.059] / +0.057 (detected from 0.01; real layer -0.000). Conclusion: for
  balance the null is a bound near 0.001; for flight and floating the evidence is the within-prompt-and-frame score,
  and a floating gain below 0.01 could go undetected.

## Files

- Phase 1: `results/phase1/` (physics_table.txt, probe_scores.csv, probe_summary.csv,
  early_prediction*.csv, run_info.json, plots, run_log.txt).
- Phase 2: `results/phase2/50steps/`, `results/phase2/1000steps/`
  (probe_scores_all_runs.csv, probe_scores_by_source.csv, early_prediction_all_runs.csv,
  physics_table.txt, run_info.json, plots), `results/phase2/checkpoint_comparison.csv`.
- Controls: `results/phase2/controls/` (`<checkpoint>_<target>_nonlinear_controls.csv` with a
  `within_prompt_r` column, `50steps_training_size.csv`).
- Label-fix analysis: `results/phase2/hovering/` (under the original rule).
- Original-label results: `results/archive_labels_v1/`.
- Cached arrays (about 9 GB, needed to rerun analyses without sampling): `data/phase2/`
  (each run folder also keeps the original-label caches as `*_v1.*`).
- Scripts: `scripts/run_phase1_local.py`, `scripts/run_phase2_local.py` (generated),
  `scripts/phase2_by_source.py`, `scripts/phase2_controls.py`, `scripts/inspect_hovering.py`,
  `scripts/early_warning_sim.py`, `scripts/intphys_vae_encode.py`, `scripts/intphys_vae_probe.py`.
- Video test: `results/intphys_vae/` (summary.txt, probe_results.csv, permutation_null.csv);
  latents in `data/intphys_vae/latents.npz` (295 MB); downloads in `vendor/wan_vae/` and
  `vendor/intphys/` (the 3.2 GB archive can be deleted once extracted).
- Early warning: `results/phase2/early_warning/`.
