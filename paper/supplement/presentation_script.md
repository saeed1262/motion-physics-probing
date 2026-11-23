# Presentation video: narration script (draft for approval)

About 640 words, so about 4.5 minutes at a calm narration pace. Each part lists what is on screen. Every number
comes from the paper. Captions on screen repeat the narration, one sentence at a time.

---

## 1. Title (about 15 s)
**On screen:** title, "Supplementary video", anonymous; the jump replaying faintly.

Recent studies of video diffusion models report that their internal activations encode physics better than their
outputs do. These models, it is said, know more than they show. We tested that claim in motion diffusion models,
where it can be checked exactly.

## 2. The question (about 20 s)
**On screen:** a floating body and a jump (from the clips); three short lines: no simulator; bodies hover; jumps fall
wrongly.

Text-to-motion models generate human movement with no physics simulator, so their typical failures are physical.
Bodies hover above the floor, and jumps do not fall as they should. If a model represented this physics inside its
layers, a probe could catch failing samples before they are finished. We set out to show exactly that.

## 3. The catch (about 30 s)
**On screen:** Figure 2, built up: noisy motion and text, the layers, the clean-motion estimate, then the two readouts.

But there is a catch. At every denoising step, a diffusion model computes its own estimate of the clean motion
from its layers. Whatever the estimate shows about the final motion has already passed through those layers. So
a probe that beats a quick readout of the estimate shows only that the layers store the information in an easier
form. The fair question is what a layer adds to the estimate, when both are read with the same care.

## 4. What we recorded (about 20 s)
**On screen:** the two models; numbers of motions; the physics labels.

We recorded every layer of two kinematic diffusion models, MDM at two checkpoints and MotionDiffuse, along
thousands of sampling trajectories. Because physics follows exactly from joint positions, we can measure balance,
flight and the fall rate of every jump, both in the final motion and in every intermediate estimate.

## 5. One jump, step by step (about 25 s)
**On screen:** clip 1, then the quarter-speed replay.

Here is one jump. From left to right: the model's estimate at zero, twenty and forty percent of denoising, and the
final motion. At zero percent the estimate is an average crouching pose. By twenty percent the jump is already in
place, and its fall rate is within about one meter per second squared of the final one. Slowed down, the estimate
at twenty percent, in teal, moves almost on top of the final motion.

## 6. The usual comparison (about 20 s)
**On screen:** Figure 3 (probe scores across denoising), then Table 2 highlighted.

Physics is readable early. A linear probe on the layers recovers the final motion's balance and flight a fifth of
the way through denoising. And against the usual baselines, the layers do look ahead of the estimate.

## 7. The fair comparison (about 30 s)
**On screen:** Figure 4, panel (a), then panel (b); then the positive control table.

But read the estimate as carefully as the layers, with the same tuned readouts on held-out prompts, and the gap
closes. For balance and flight, adding a layer to the estimate gains at most one hundredth in R squared. A positive
control, in which we hide a known signal in the layers, shows that the protocol would have found a gain that small.

## 8. The one exception (about 45 s)
**On screen:** Figure 8 (b to d), then clip 3 (the three jumps).

There is one exception: how fast a jump falls. A fifth of the way through denoising, the layers add up to eight
hundredths in R squared to the best readout of the estimate, and on the fifty-step model the gain survives every
control we tried. But it does not last. By sixty percent of denoising it has fallen below two hundredths,
because the estimate catches up. On MotionDiffuse the same gain is small and not robust. Here are three of the jumps where the layer
helps most, chosen for that reason. The estimate alone misjudges how fast they fall, and adding the layer brings
the prediction closer.

## 9. Floating and early warning (about 35 s)
**On screen:** clip 2 (the floating walk), then Figure 10.

The most common failure is floating, and the estimate shows it early too. At twenty percent of denoising this
estimate already floats where the final motion does. A simple rule on the estimate, with no training, can abort
samples that will float, and it does as well as a probe on the layers. The payoff is small, though. Aborting saves
under a tenth of the compute, because floating is rare and concentrated in a few prompts.

## 10. A video model (about 25 s)
**On screen:** Table 6 (IntPhys readouts), with the change-feature row highlighted.

A related readout issue appears in video. For the Wan video model, a linear probe on the clean latent is at chance
for physical plausibility, which was read as evidence that the latent lacks it. One fixed nonlinearity, the size of
the change from frame to frame, makes plausibility readable from that same latent.

## 11. Five checks (about 30 s)
**On screen:** the five checks, appearing one by one.

For anyone probing a generator for hidden knowledge, our results suggest five checks. Read the model's own estimate
as carefully as its layers, and run a positive control. Score within prompts, and vary the amount of training
data. Try simple transforms before declaring something absent. Look at what your target actually measures. And
for early pruning, compare with the model's decoded preview.

## 12. Close (about 10 s)
**On screen:** the title again; "every body, path and number is computed from the recorded outputs".

In short, these models mostly show what they know. The one thing their layers know first, how fast a jump will
fall, their estimate shows later in denoising.
