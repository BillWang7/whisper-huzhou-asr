"""Transcribe a whole test set with one model, using the demo's decoding settings.

    python -m scripts.run_test_set baseline                  # news test set -> examples/test_set/baseline.csv
    python -m scripts.run_test_set finetuned                 # news test set -> examples/test_set/finetuned.csv
    python -m scripts.run_test_set finetuned --set real      # real speakers -> examples/real_speakers/finetuned.csv

Run from the repo root. Uses an NVIDIA or Apple GPU if available (the 1,705 news clips take about 35 minutes
on an M4 MacBook's GPU, the 828 real-speaker clips about 20; much longer on CPU). Results are appended as
they are made, so an interrupted run picks up where it stopped. Score the results with `python -m scripts.score`.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.decoding import generate, pick_device

MODELS = {
    "baseline": ("openai/whisper-medium", "baseline_prediction"),
    "finetuned": ("BillWang7/whisper-medium-ch-hz", "finetuned_prediction"),
}
SETS = {"news": Path("examples/test_set"), "real": Path("examples/real_speakers")}

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=MODELS)
parser.add_argument("--set", choices=SETS, default="news", help="news broadcasts (default) or real speakers")
parser.add_argument("--out", help="output CSV (default: <set folder>/<model>.csv)")
parser.add_argument("--limit", type=int, help="only the first N clips, for a quick test")
args = parser.parse_args()

model_id, column = MODELS[args.model]
test_dir = SETS[args.set]
out_csv = Path(args.out or test_dir / f"{args.model}.csv")

device = pick_device()
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()

clips = sorted(int(f.stem) for f in (test_dir / "audio").glob("*.mp3"))
if args.limit:
    clips = clips[: args.limit]

done = set()
if out_csv.exists():
    with open(out_csv, encoding="utf-8") as f:
        done = {int(r["clip"]) for r in csv.DictReader(f)}
new_file = not out_csv.exists()
out = open(out_csv, "a", newline="", encoding="utf-8")
writer = csv.writer(out)
if new_file:
    writer.writerow(["clip", column])

todo = [c for c in clips if c not in done]
print(f"{model_id} on {device}: {len(clips)} clips, {len(done)} already done, {len(todo)} to go", flush=True)
start = time.time()
for n, clip in enumerate(todo, 1):
    audio, sr = torchaudio.load(str(test_dir / "audio" / f"{clip}.mp3"))
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.mean(dim=0)
    inputs = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        ids = generate(model, inputs.input_features.to(device), inputs.attention_mask.to(device))
    writer.writerow([clip, processor.batch_decode(ids, skip_special_tokens=True)[0]])
    out.flush()
    if n % 50 == 0 or n == len(todo):
        per = (time.time() - start) / n
        print(f"{n}/{len(todo)}  {per:.2f}s/clip  ~{per * (len(todo) - n) / 60:.0f} min left", flush=True)
out.close()

# Keep the file in clip order (a resumed run can append out of order).
with open(out_csv, encoding="utf-8") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: int(r["clip"]))
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["clip", column])
    writer.writerows([r["clip"], r[column]] for r in rows)
print("done:", out_csv, flush=True)
