"""Voice a list of sentences with Kokoro (offline). Runs in vendor/kokoro-venv:

    vendor/kokoro-venv/bin/python scripts/tts_kokoro.py <sentences.json> <out_dir> [voice] [speed]

sentences.json: [{"id": "s03_2", "text": "..."}]. Writes <out_dir>/<id>.wav (24 kHz mono) and <out_dir>/durations.json.
"""
import json, os, sys
import numpy as np
import soundfile as sf
from kokoro import KPipeline

src, out = sys.argv[1], sys.argv[2]
voice = sys.argv[3] if len(sys.argv) > 3 else "am_michael"
speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
os.makedirs(out, exist_ok=True)
pipe = KPipeline(lang_code="a")
dur = {}
for item in json.load(open(src)):
    chunks = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(item["text"], voice=voice, speed=speed)]
    audio = np.concatenate(chunks).astype(np.float32)
    sf.write(os.path.join(out, f"{item['id']}.wav"), audio, 24000)
    dur[item["id"]] = len(audio) / 24000
    print(f"{item['id']}: {dur[item['id']]:.1f} s", flush=True)
json.dump(dur, open(os.path.join(out, "durations.json"), "w"), indent=1)
print(f"total {sum(dur.values()):.1f} s over {len(dur)} sentences")
