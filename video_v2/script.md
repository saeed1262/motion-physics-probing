# Supplementary video, revised: narration script (draft for approval)

A revised version of the anonymous supplementary video (`paper/supplement/presentation.mp4`, 5:40), for a TMLR
revision. It stays anonymous: "we", no names, no arXiv id, no link to the project page. It uses the paper's terms
(clean-motion estimate, layers, R²) with plain glosses, so it matches what reviewers read. What is new, taken from the
project page: animated charts in place of the dense paper figures, the two-readers explanation, the airborne-label fix
(Appendix) and the pilot that flipped (Section 4.4). Every number comes from the paper. About 820 words, so about
6.5 minutes at the narration pace of the submitted video (726 words, 5:40). Captions repeat the narration one sentence at a time. Items
marked NEW are animated slides the submitted video does not have.

---

## 1. Floating (about 25 s)
**On screen:** the floating-walk clip, full size, body red while it floats; then the title card ("Supplementary video ·
anonymous submission").

Ask a motion diffusion model for a person walking forward slowly and carefully, and watch closely. Most samples are
fine. But in some, in the middle of a step, both feet leave the floor and the whole body glides a few centimeters up,
for half a second, before it sets down again. Nothing pulls it back down, because nothing in the model enforces the
ground.

## 2. The question (about 25 s)
**On screen:** three short lines appearing: "no physics simulator", "physics learned from motion capture, never
enforced", "do they know more than they show?"

Kinematic motion generators output joint positions directly, so their physics is learned from motion capture and
never enforced. Video diffusion models were recently reported to know more physics in their activations than they show
in their outputs. If motion models did too, a probe on the layers could catch a sample like that one early. We set out
to show exactly that. It is not what we found.

## 3. The clean-motion estimate (about 35 s)
**On screen:** the jump clip at four stages in sync (0%, 20%, 40%, final), then the quarter-speed replay with the
estimate at 20% overlaid in teal.

A diffusion model builds a motion out of noise in many small steps, and at every step it computes a clean-motion
estimate: its running guess of the final motion. Here is one jump at four points of denoising. At zero percent the
estimate is an average crouch. By twenty percent the jump is in place, and it falls at nine and a half meters per
second squared, close to gravity. Slowed down, the estimate at twenty percent moves almost on top of the final motion.

## 4. Measuring physics (about 35 s)
**On screen:** Figure 3, one panel at a time: balance, then the fitted parabola, then the floating walk.

Because the output is joint positions, physics can be computed exactly, for the final motion and for every estimate:
balance from the extrapolated center of mass, and for flight, a parabola fitted to the center of mass in the air. A real
jump falls at nine point eight one; a body that falls slower than two is floating. Our first definition of "airborne"
looked only at the feet and counted crawling and hovering as flight; the corrected labels separate them.

## 5. The catch (about 35 s)
**On screen:** NEW, the two readers: "reader A sees the estimate", "reader B sees the estimate and a layer", both
pointing to "the final motion's physics"; then Figure 2 on the recorded jump.

Here is the catch. The estimate is computed from the model's layers, so it cannot carry information about the outcome
that the layers and the noisy input lack. A probe that beats a quick readout of the estimate only shows that the layers
store the information in an easier form. So for every target we train two readers, one on the estimate and one on the
estimate and a layer, both tuned the same way. The gap between them is what the layer adds.

## 6. The advantage disappears (about 40 s)
**On screen:** NEW, the comparison chart: bars for the usual comparison, then on the same axis the fair comparison's
dots and intervals; the fall-rate row stays above zero, highlighted.

Against the usual baselines, the layers look ahead of the estimate on almost every target. Read the estimate with the
same care, and the advantage disappears. For balance and flight, on all three models, adding a layer gains at most
one hundredth, one point on a hundred-point scale. A positive control, in which we hide a small known
signal in the layer, shows that the protocol would have found a gain that small.

## 7. Small data can flip the answer (about 20 s)
**On screen:** NEW, two numbers side by side: "pilot, 64 prompts: the explicit label wins, 0.832 against 0.814";
"25 training prompts: the explicit label beats every trained readout".

Our pilot study, with sixty-four prompts, pointed the other way, and the reason was data. With too few training
examples, a high-dimensional probe and a one-number baseline can trade places. Only a lead that persists as the
training set grows is evidence about the model.

## 8. The exception (about 45 s)
**On screen:** NEW, the fall-rate chart drawing step by step, the shaded gap between "the estimate" and "the estimate
plus a layer" narrowing from 20% to 80%; then the clip of the jumps where the layer helps most (labelled as selected).

There is one exception: how fast a jump falls. A fifth of the way through denoising, MDM's layers add about eight
hundredths in R squared that the estimate does not yet show, and on the fifty-step model the gain survives every
control we applied. But it fades: by sixty percent of the way it is under two hundredths. A real case of a model
knowing more than it shows, but a small and short-lived one.

## 9. What actually goes wrong (about 25 s)
**On screen:** the floating walk again, the estimate at 20% above the final motion, floating in the same places.

What does go wrong? Mostly not the jumps: when MDM's bodies fall, they fall at close to the right rate. The common
failure is floating, and on test captions the original thousand-step checkpoint floats about twice as often as the
fifty-step one. The estimate at twenty percent already floats where the final motion does.

## 10. Early warning (about 35 s)
**On screen:** NEW, the early-warning chart drawing its three curves as the number of tries grows: finish then filter,
abort at 20%, a perfect detector.

That makes the estimate an early warning, with no training at all. If it floats at twenty percent, the final motion
floats in about eighty percent of cases. But the payoff is small. Floating affects only nine to eighteen percent of
samples and concentrates in a few prompts, so aborting early saves under a tenth of the compute. Even a perfect
detector would save little.

## 11. A video model (about 25 s)
**On screen:** NEW, bars for the readouts of the Wan 2.1 clean latent: every linear one near 50%, then one rising to
78% after the frame-to-frame change.

In video, a linear probe at chance on Wan two point one's clean latent was read as evidence that plausibility is
absent there. One fixed transformation, the size of the change between frames, lifts a linear readout to seventy-eight
percent pairwise accuracy on IntPhys. A probe at chance is weak evidence of absence.

## 12. Five checks (about 30 s)
**On screen:** the five checks appearing one by one (as in the submitted video).

Before concluding that a generator knows more than it shows, we suggest five checks. Read its own estimate as
carefully as its layers, and run a positive control. Score within prompts, and vary the amount of training data. Try
simple transforms before declaring something absent. Look at what a target's variance is made of. And for early
pruning, compare with scoring the decoded preview.

## 13. End (about 10 s)
**On screen:** the closing card of the submitted video ("Every body, path and number is computed from the recorded
outputs of the models").

Every body, path and number in this video is computed from the recorded outputs of the models.
