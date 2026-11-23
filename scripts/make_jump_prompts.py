"""Prompt list for the jump-heavy run: more flight, to measure the gravity comparison precisely.

The 14 curated flight prompts from phase 2 (the ones its group table calls "curated: flight")
plus one caption from every HumanML3D *test* motion that has a caption containing jump, hop, leap
or skip as a word (any tense), 3 to 20 words long. The caption per motion is drawn with seed 0
among its matching captions, as phase 2 draws its test captions. Duplicates are dropped.

    .venv/bin/python scripts/make_jump_prompts.py

Writes prompts/jump_heavy.txt, one prompt per line, for
`run_phase2_local.py --prompts-file prompts/jump_heavy.txt --tag jump_heavy`.
"""
import os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HML = os.path.join(ROOT, "vendor", "HumanML3D", "HumanML3D")
CURATED_FLIGHT = [
    "a person jumps up in place.", "a person jumps forward.", "a person does a long jump forward.",
    "a person hops on their left leg.", "a person hops on their right leg.", "a person jumps over an obstacle.",
    "a person skips forward.", "a person does jumping jacks.", "a person jumps up and spins in the air.",
    "a person leaps to the side.", "a person runs and jumps.", "a person jumps backward.",
    "a person jumps with both feet three times.", "a person jumps and lands on one foot.",
]
WORD = re.compile(r"\b(jump|jumps|jumped|jumping|hop|hops|hopped|hopping|"
                  r"leap|leaps|leaped|leapt|leaping|skip|skips|skipped|skipping)\b")

ids = [l.strip() for l in open(os.path.join(HML, "test.txt")) if l.strip()]
rng = np.random.default_rng(0)
out, seen = list(CURATED_FLIGHT), set(CURATED_FLIGHT)
for i in sorted(ids):
    path = os.path.join(HML, "texts", i + ".txt")
    if not os.path.exists(path):
        continue
    caps = [l.split("#")[0].strip() for l in open(path) if l.strip()]
    caps = [c for c in caps if 3 <= len(c.split()) <= 20 and WORD.search(c.lower())]
    if not caps:
        continue
    cap = caps[rng.integers(len(caps))]
    cap = cap if cap.endswith(".") else cap + "."
    if cap not in seen:
        out.append(cap); seen.add(cap)
os.makedirs(os.path.join(ROOT, "prompts"), exist_ok=True)
with open(os.path.join(ROOT, "prompts", "jump_heavy.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
print(f"{len(out)} prompts ({len(CURATED_FLIGHT)} curated + {len(out) - len(CURATED_FLIGHT)} test captions) "
      "-> prompts/jump_heavy.txt")
