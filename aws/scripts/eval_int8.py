"""Compare the 8-bit Lambda model with the published float32 results on the same clips.

Runs inside the transcriber image with the repo mounted at /repo:
  docker run --rm --entrypoint python -v "$PWD:/repo" -w /repo -e QUANTIZE_INT8=1 whisper-lambda \
      aws/scripts/eval_int8.py 200 100
Arguments: number of news clips, number of real-speaker clips (the first N of each test set).
The float32 predictions come from examples/*/finetuned.csv, so only the int8 model is run here.
"""
import csv
import statistics
import sys
import time

sys.path.insert(0, "/var/task")  # transcriber app.py
sys.path.insert(0, "/repo")      # src.scoring

import app  # noqa: E402
from src.scoring import score  # noqa: E402

SETS = [("News broadcasts", "examples/test_set"), ("Real speakers", "examples/real_speakers")]


def rows(path):
    """clip number -> row. The news text.csv has no clip column; its row order is the clip number."""
    with open(path, encoding="utf-8") as f:
        return {int(r.get("clip", n)): r for n, r in enumerate(csv.DictReader(f))}


for (name, folder), n in zip(SETS, map(int, sys.argv[1:3])):
    text, fp32 = rows(f"{folder}/text.csv"), rows(f"{folder}/finetuned.csv")
    if n == 0:  # e.g. `eval_int8.py 0 100` checks only the real speakers
        continue
    text_col = "Text"  # real_speakers/text.csv also has a "set" column, so don't guess
    clips = sorted(text)[:n]
    c32, c8, same, secs = [], [], 0, []
    for i in clips:
        ref, old = text[i][text_col], fp32[i]["finetuned_prediction"]
        t = time.perf_counter()
        new = app.transcribe(f"{folder}/audio/{i}.mp3")["text"]
        secs.append(time.perf_counter() - t)
        c32.append(score(ref, old)[0])
        c8.append(score(ref, new)[0])
        same += old == new
    print(f"{name}: n={len(clips)}  float32 CER {statistics.mean(c32):.1f}%  int8 CER {statistics.mean(c8):.1f}%  "
          f"identical outputs {same}/{len(clips)}  ({statistics.mean(secs):.1f}s/clip here)", flush=True)
