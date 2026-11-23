# Research Plan: Do Motion Models Know About Balance and Gravity?

Nov 15, 2026 · @Saeed Ghorbani

## Summary

We test whether a text-to-motion diffusion model (MDM) internally represents balance and gravity, and whether those internal signals flag physical failures before the output shows them. No prior work has looked inside human motion generators for physics.

**Hypotheses**

- **H1 (readable):** dynamic balance, floating and flight are linearly decodable from MDM's middle layers.
- **H2 (knows more than it shows):** early in denoising, activations predict the final motion's physics better than the model's own current guess of the clean motion.
- **H3 (early warning):** early-step activations predict which samples will float or lose balance, beyond what the prompt alone predicts.

Physical artifacts (floating, sliding, falling over) are the main quality problem in generated character animation. Knowing whether the physics is missing inside the model, or present but lost at the output, tells the field which part to fix.

## Positioning

Our novelty is the first internal analysis of human motion generators, with body-specific physics (center of mass, support polygon, flight) and an early-warning payoff. Closest work either probes a different domain (video) or fixes physics from the outside.

| Work | What it does | How we differ |
| --- | --- | --- |
| [Does Physics Live in the Activations?](https://arxiv.org/abs/2610.03154) (2026) | Probes video DiTs for simulator physics quantities | Our method template; we target human motion and body balance, not rigid objects in video, and record along the model's own sampling path rather than from noised real videos |
| [The Invisible Hand of Physics](https://arxiv.org/abs/2606.05328) (2026) | Video diffusion models encode plausibility not visible in outputs | Same "knows more than it shows" question, asked for character animation |
| [Seeking Physics in Diffusion Noise](https://arxiv.org/abs/2603.14294) (2026) | Uses probes in video DiTs to prune implausible samples early in sampling | Closest to our early-warning test; we do it for human motion, with within-prompt controls that rule out prompt difficulty |
| [SAEs in VLA models](https://arxiv.org/abs/2603.19183) (2026) | Steerable motion-primitive features in robot policies | Robot control policies, not text-to-motion generators |
| [MotionCLR](https://arxiv.org/abs/2410.18977) (2024) | Attention analysis in motion diffusion for editing | Text-to-motion correspondence, not physics |
| [Think Before You Move](https://arxiv.org/abs/2512.24100) (2025) | Latent reasoning for text-to-motion; contrasts semantic and reconstruction latents | A generation method, not an analysis of model internals or physics |
| [PhysDiff](https://arxiv.org/abs/2212.02500) (2023), [SimDiff](https://arxiv.org/abs/2509.20927) (2025), [PhysibleMDM](https://arxiv.org/abs/2610.30500) (2026) | Fix physics from outside: simulator projection, conditioning, noise optimization | We explain where physics fails inside the model; our early warning is a complement, not a competitor |
| Physics-based control: [MaskedMimic](https://arxiv.org/abs/2409.14393) (2024), [CLoSD](https://arxiv.org/abs/2410.03441) (2024), [MotionBricks](https://arxiv.org/abs/2604.24833) (2026) | A simulator enforces physics, or a kinematic model feeds a robot controller | Different question: we study kinematic generators that can break physics. Our probes could later be applied to the diffusion planners inside hybrids like CLoSD (future work) |

One honest risk: video-model physics probing is moving monthly, so a motion-specific preprint could appear. Re-check arXiv before each submission.

## Method

We record MDM's activations while it generates motions, compute physics labels from the final motions, and fit linear probes from activations to labels. All of it runs on a free Colab T4.

**Model and data**

- MDM, 50-step encoder checkpoint (8 transformer layers, 512 dimensions, CLIP text encoder).
- 64 prompts stressing jumps and balance, 4 samples each (256 motions, 120 frames at 20 fps).
- Only HumanML3D's normalization files are needed; no motion-capture data or AMASS registration.

**Physics labels** (from joint positions)

- **Dynamic balance:** signed distance from the extrapolated center of mass (Hof 2005) to the support polygon of the feet in contact. Probed as a continuous quantity: a real-motion check showed it also goes negative during normal stepping.
- **Static balance:** the same with the plain center of mass, for comparison.
- **Floating:** height of the lowest foot. **Flight:** no foot in contact.
- **Per-sample failure** measures: gravity error during flight (main test; the fitted fall rate must be 9.81 m/s², so it needs no calibration), share of frames floating, and an exploratory off-balance share to be calibrated on real motion.

**Probes and controls**

- Ridge regression per layer and per recorded denoising step; train and test split by prompt, so probes generalize to unseen actions.
- **Baselines:** probes on the noisy input, probes on the model's current clean-motion guess, and labels computed directly from that guess.
- **Text only:** for failure prediction, a probe on the prompt's CLIP embedding, since some prompts fail more often.
- **Within-prompt ranking:** among samples of the same prompt, does the probe rank the failing ones higher? Only sample-specific knowledge can do this.

**Later phases add** causal steering along probe directions and a recent second model (Kimodo, 2026) to show generality.

## Experiments

Five phases, each ending in a go/no-go check, so we stop early if the core signal isn't there. Thresholds are starting targets; we adjust them once we see real numbers.

1. **Feasibility** (current notebook, about 1 week)
   - Go if dynamic balance reaches R² ≥ 0.3 at some layer, or flight reaches AUC ≥ 0.8, and either beats the clean-motion-guess baseline at an early step.
   - No-go: switch to the fallbacks in the next section.
2. **Scale and robustness** (about 2 weeks)
   - About 1,000 motions using HumanML3D's public test captions, 3 seeds, plus the original 1,000-step MDM.
   - Go if the main results hold across seeds and checkpoints.
3. **Causal tests** (about 2–3 weeks)
   - Steer along probe directions and patch activations between floating and grounded samples; measure output balance, floating and FID.
   - Go if outputs move in the predicted direction with modest FID cost. This turns "readable" into "used by the model."
4. **Early warning** (about 2 weeks)
   - Flag likely failures at early steps and resample them. Compare compute against "finish, then filter" at the same final failure rate.
   - Go if it beats the text-only baseline and within-prompt ranking stays above ρ = 0.2 by the 40% mark of denoising.
5. **Generality** (about 2–3 weeks)
   - Repeat phases 1 and 4 on Kimodo (NVIDIA, 2026), a 282M-parameter motion diffusion model built to avoid floating and foot skating, using its 22-joint SMPL-X checkpoint. Labels need small changes for its 30 fps output. MoMask (tokenized) stays optional.
   - Go if at least one main finding replicates; a clear difference between MDM and Kimodo is also a finding, since it shows how model design changes what is represented inside.

## Risks and fallbacks

The biggest risk is a weak or absent balance signal; even then the paper survives as a diagnostic or negative result.

| Risk | Fallback |
| --- | --- |
| Balance-failure label flags normal stepping (seen on a real clip) | Gravity test becomes the main failure measure; calibrate balance against real HumanML3D motion once AMASS access arrives |
| Early warning was already shown for video (Tang et al., 2026) | Present it as the motion version with within-prompt controls; lead with the balance and gravity findings |
| Balance isn't decodable, but flight and floating are | Paper becomes "motion models track contact and flight but not balance"; a clean negative result with a clear lesson for training |
| Probes only match the clean-motion-guess baseline | Story shifts from hidden knowledge to how physics is decoded across steps; still a first internal analysis |
| Steering creates artifacts (known issue in diffusion models) | Use probe-based guidance during sampling instead of editing activations directly |
| A motion-probing preprint appears first | Lean on what's ours: body balance and gravity, recording along the sampling path, within-prompt tests, MDM vs. Kimodo comparison |
| Kimodo may not run on the free T4 (its supported GPUs start at the Ampere generation) | Try the free T4 first; otherwise use a paid Colab L4 or A100 for phase 5 only, or fall back to MoMask |
| Notebook breaks on first real run | Fix in place; only the model-loading cells are untested on real hardware |
| Google Drive blocks the checkpoint download | Copy it to your own Drive and load from there (the notebook explains how) |
| Colab time limits | Phase 2 saves every batch to Drive and resumes after a disconnect |

## Target venues

Primary target: TMLR, a journal with rolling submissions and no travel, aiming to submit by February 5, 2027. The conferences below require presenting in person, so they apply only if travel becomes possible; **Graphics Interface 2027 in Winnipeg** accepts a recorded video only for documented visa or health reasons. Dates marked "est." follow last year's pattern and must be confirmed when the call is posted.

| Venue | Where and when | Deadline | In person? | Fit and notes |
| --- | --- | --- | --- | --- |
| [TMLR](https://jmlr.org/tmlr/index.html) | Online journal | Rolling; we aim for Feb 5, 2027 | No | **Primary.** Decision aimed at about 9 weeks ([FAQ](https://jmlr.csail.mit.edu/tmlr/faq.html)); free, open access. Strong papers can be invited to present at NeurIPS, ICML or ICLR (optional). Has paused new submissions over past December holidays |
| [Graphics Interface 2027](https://conferences.graphicsinterface.org/2027/submissions/) | Winnipeg, Canada; June 15–18, 2027 | Feb 5, 2027 (official) | Yes; recorded video only for visa or health reasons | Full archival paper in ACM DL, CORE B |
| [Eurographics 2027 short papers](https://eg2027.isti.cnr.it/call-for-short-papers/) | Lucca, Italy; May 10–14, 2027 | Jan 22, 2027 (official) | Yes; exceptions need a documented visa or health case | 4 pages; ML work related to graphics explicitly welcome |
| ICLR 2027 workshops ([call](https://iclr.cc/Conferences/current/CallForWorkshops)) | San Francisco, USA; April 29–30, 2027 | Early Feb 2027 (est.) | Usually; varies by workshop | Usually non-archival |
| CVPR 2027 workshops ([dates](https://cvpr.thecvf.com/Conferences/2027/Dates)) | Seattle, USA; June 20–21, 2027 | Mar 2027 (est.) | Yes; no fully virtual workshops | Workshop list not yet published |
| [SCA 2027](https://computeranimation.org/) | Not announced | Early Apr 2027 (est.) | Expected | Best topical fit; papers appear in Computer Graphics Forum |
| SIGGRAPH 2027 Posters | Anaheim, USA; Aug 8–12, 2027 | Late Apr 2027 (est.) | Expected | Short abstract in ACM DL |
| [MIG 2027](https://mig.siggraph.org/2026/papers.htm) | Not announced | Aug 2027 (est.) | Yes (2026 rule) | CORE C, motion-focused |

**Ruled out for 2027:** CVPR main (deadline Nov 16, 2026), ICLR and AAAI main tracks (passed), ICML 2027 (Rio de Janeiro), ICCV 2027 (Hong Kong). **Watch list:** NeurIPS 2027 (Europe, deadline around May 2027, est.) and ICML 2028 (Eastern US), both in person.

TMLR doesn't accept work that overlaps previously published papers, so it should get the first archival version. Non-archival workshop talks are usually fine, but check each workshop's rules.

## Timeline

&#91;embedded content: plan timeline · 5 phases and the TMLR submission\]

Phases 1–4 finish by mid-January, leaving three weeks to write before the February 5 target. TMLR has no fixed deadline, so a slip moves the date rather than costing a whole review cycle. Phase 5 runs during review and goes into the revision.

## Status and next steps

Literature checks, both experiment notebooks and the paper skeleton are done; nothing has run on a GPU yet. Everything downstream waits on its first results.

- [x] Choose direction and run three literature checks
- [x] Build the feasibility notebook (motion\_balance\_probe.ipynb); analysis code tested on synthetic data
- [x] Draft related work, bibliography (19 verified references) and paper skeleton in TMLR format
- [x] Build the phase 2 notebook (scale and robustness, saves to Drive and resumes)
- [x] Set a monthly arXiv reminder through April 2027
- [x] Add a calibration-free gravity label after a real-clip check showed the balance-failure label flags normal stepping
- [ ] Run the phase 1 notebook on a Colab T4 and fix any loading errors
- [ ] Review the results and make the phase 1 go/no-go call
- [ ] Register for AMASS, SMPL+H and DMPL to calibrate labels on real motion
- [ ] Upload the paper skeleton to Overleaf
