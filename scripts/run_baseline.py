"""Baseline openai/whisper-medium on the Huzhou test set, forced to Chinese.

Same decoding settings as the fine-tuned demo (language="zh", task="transcribe").
Appends to the output CSV and resumes where it left off.
Usage: python run_baseline.py AUDIO_DIR OUT_CSV [LIMIT]
"""
import csv
import os
import sys
import time

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

audio_dir, out_csv = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device).eval()

clips = sorted(int(f[:-4]) for f in os.listdir(audio_dir) if f.endswith(".mp3"))
if limit:
    clips = clips[:limit]

done = set()
if os.path.exists(out_csv):
    with open(out_csv, encoding="utf-8") as f:
        done = {int(r["clip"]) for r in csv.DictReader(f)}
new_file = not os.path.exists(out_csv)
out = open(out_csv, "a", newline="", encoding="utf-8")
w = csv.writer(out)
if new_file:
    w.writerow(["clip", "baseline_prediction"])

todo = [c for c in clips if c not in done]
print(f"device={device} total={len(clips)} done={len(done)} todo={len(todo)}", flush=True)
t0 = time.time()
for n, c in enumerate(todo, 1):
    audio, sr = torchaudio.load(os.path.join(audio_dir, f"{c}.mp3"))
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.mean(dim=0)
    inputs = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        ids = model.generate(
            inputs.input_features.to(device),
            attention_mask=inputs.attention_mask.to(device),
            language="zh",
            task="transcribe",
        )
    w.writerow([c, processor.batch_decode(ids, skip_special_tokens=True)[0]])
    out.flush()
    if n % 25 == 0 or n == len(todo):
        el = time.time() - t0
        print(f"{n}/{len(todo)} {el/n:.2f}s/clip eta {el/n*(len(todo)-n)/60:.1f} min", flush=True)
out.close()
print("DONE", flush=True)
