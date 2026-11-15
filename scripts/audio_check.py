"""Find real-speaker clips that are probably bad recordings (silent, cut off, too quiet, clipped, noisy, or the
speaker said much less/more than the sentence), so they can be checked by ear.

    .venv/bin/python -m scripts.audio_check

Measures each clip with a simple energy-based speech detector (no model), compares the amount of speech with the
length of the sentence, and joins the fine-tuned model's CER. Writes data/audio_check.csv (every clip, worst first)
and prints a summary. The flags are hints, not verdicts: listen before dropping anything.
"""
import csv
import shutil
import statistics
from pathlib import Path

import numpy as np
import torchaudio

from src.scoring import normalize, score

REAL = Path("examples/real_speakers")
OUT = Path("data/audio_check.csv")
REVIEW = Path("data/audio_review")  # copies of the clips to listen to, plus review.csv
BAD_AUDIO = {"NO_SPEECH", "TOO_LITTLE_SPEECH", "EXTRA_SPEECH", "TOO_QUIET", "CLIPPED", "NOISY", "OVER_30S"}
SR = 16000
FRAME = 400  # 25 ms
HOP = 160    # 10 ms


def frame_db(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[np.clip(idx, 0, len(x) - 1)] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def measure(path: Path) -> dict:
    audio, sr = torchaudio.load(str(path))
    raw = audio.numpy()
    clipped = float(np.mean(np.abs(raw) > 0.99))
    x = torchaudio.functional.resample(audio.mean(0), sr, SR).numpy()
    db = frame_db(x)
    floor, loud = np.percentile(db, 10), np.percentile(db, 98)
    # A frame is speech if it is well above the background and not far below the loudest part
    threshold = max(floor + 12, loud - 35)
    speech = db > threshold
    # Smooth: fill gaps < 200 ms, drop blips < 100 ms
    s = speech.copy()
    runs, start = [], None
    for i, v in enumerate(np.append(s, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 20:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if r[1] - r[0] >= 10]
    n = len(db)
    voiced = sum(b - a for a, b in merged) * HOP / SR
    return {
        "seconds": round(len(x) / SR, 2),
        "sample_rate": sr,
        "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(raw)) + 1e-12)), 1),
        "loudness_dbfs": round(float(loud), 1),
        "snr_db": round(float(loud - floor), 1),
        "clipped_pct": round(100 * clipped, 3),
        "speech_seconds": round(voiced, 2),
        "lead_silence_s": round(merged[0][0] * HOP / SR, 2) if merged else None,
        "trail_silence_s": round((n - merged[-1][1]) * HOP / SR, 2) if merged else None,
    }


def main():
    rows = sorted(csv.DictReader(open(REAL / "text.csv", encoding="utf-8")), key=lambda r: int(r["clip"]))
    pred = {r["clip"]: r["finetuned_prediction"] for r in csv.DictReader(open(REAL / "finetuned.csv", encoding="utf-8"))}
    base = {r["clip"]: r["baseline_prediction"] for r in csv.DictReader(open(REAL / "baseline.csv", encoding="utf-8"))}

    out = []
    for n, r in enumerate(rows):
        m = measure(REAL / "audio" / f"{r['clip']}.mp3")
        chars = len(normalize(r["Text"]))
        out.append({"clip": int(r["clip"]), "set": r["set"], "chars": chars, **m,
                    "cer_finetuned": round(score(r["Text"], pred[r["clip"]])[0], 1),
                    "cer_baseline": round(score(r["Text"], base[r["clip"]])[0], 1),
                    "Text": r["Text"], "finetuned": pred[r["clip"]], "baseline": base[r["clip"]]})
        if (n + 1) % 100 == 0:
            print(f"measured {n + 1}/{len(rows)}", flush=True)

    # Typical speaking rate (characters per second of speech) per set, from the clips the model got mostly right
    rate = {}
    for s in {o["set"] for o in out}:
        good = [o["chars"] / o["speech_seconds"] for o in out
                if o["set"] == s and o["speech_seconds"] > 0.5 and o["cer_finetuned"] <= 30]
        rate[s] = statistics.median(good)
    for o in out:
        expected = o["chars"] / rate[o["set"]]
        o["expected_speech_s"] = round(expected, 2)
        o["speech_vs_expected"] = round(o["speech_seconds"] / expected, 2) if expected else None
        flags = []
        if o["speech_seconds"] < 0.4 or o["snr_db"] < 8:
            flags.append("NO_SPEECH")
        elif o["speech_vs_expected"] < 0.45:
            flags.append("TOO_LITTLE_SPEECH")
        # Short phrases (a few characters) are often said slowly, so only judge "too much speech" on longer ones
        if o["chars"] >= 8 and o["speech_vs_expected"] and o["speech_vs_expected"] > 2.2:
            flags.append("EXTRA_SPEECH")
        if o["peak_dbfs"] < -26:
            flags.append("TOO_QUIET")
        if o["clipped_pct"] > 0.05:
            flags.append("CLIPPED")
        if 8 <= o["snr_db"] < 18:
            flags.append("NOISY")
        if o["lead_silence_s"] is not None and o["lead_silence_s"] < 0.03:
            flags.append("CUT_START")
        if o["trail_silence_s"] is not None and o["trail_silence_s"] < 0.03:
            flags.append("CUT_END")
        if o["seconds"] > 30:
            flags.append("OVER_30S")
        o["flags"] = " ".join(flags)

    out.sort(key=lambda o: (-o["cer_finetuned"], o["clip"]))
    OUT.parent.mkdir(exist_ok=True)
    cols = ["clip", "set", "flags", "cer_finetuned", "cer_baseline", "seconds", "speech_seconds", "expected_speech_s",
            "speech_vs_expected", "chars", "peak_dbfs", "loudness_dbfs", "snr_db", "clipped_pct", "lead_silence_s",
            "trail_silence_s", "sample_rate", "Text", "finetuned", "baseline"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    print("speaking rate (chars/s of speech):", {k: round(v, 2) for k, v in rate.items()})
    names = ["NO_SPEECH", "TOO_LITTLE_SPEECH", "EXTRA_SPEECH", "TOO_QUIET", "CLIPPED", "NOISY", "CUT_START",
             "CUT_END", "OVER_30S"]
    for name in names:
        hit = [o for o in out if name in o["flags"].split()]
        if hit:
            print(f"{name:18} {len(hit):4} clips  mean CER {statistics.mean(o['cer_finetuned'] for o in hit):5.1f}%")
    clean = [o for o in out if not o["flags"]]
    flagged = [o for o in out if o["flags"]]
    print(f"no flags           {len(clean):4} clips  mean CER {statistics.mean(o['cer_finetuned'] for o in clean):5.1f}%")
    print(f"any flag           {len(flagged):4} clips  mean CER {statistics.mean(o['cer_finetuned'] for o in flagged):5.1f}%")
    print(f"wrote {OUT}")

    # Listening list: (1) audio problems, (2) both models hear the same thing but it isn't the sentence
    # (speaker probably said other words), (3) every other clip the fine-tuned model got 100%+ wrong
    review = []
    for o in out:
        ft_vs_base = score(o["finetuned"], o["baseline"])[0] if o["finetuned"] else 100
        if BAD_AUDIO & set(o["flags"].split()):
            review.append(("1_audio_problem", o["flags"], o))
        elif o["cer_finetuned"] >= 60 and ft_vs_base <= 40:
            review.append(("2_said_other_words", f"models agree with each other ({ft_vs_base:.0f}% apart)", o))
        elif o["cer_finetuned"] >= 100:
            review.append(("3_model_fully_wrong", "audio measures look normal", o))
    if REVIEW.exists():
        shutil.rmtree(REVIEW)
    REVIEW.mkdir(parents=True)
    with open(REVIEW / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "clip", "why", "cer_finetuned", "seconds", "speech_seconds", "expected_speech_s",
                    "peak_dbfs", "snr_db", "Text", "finetuned", "baseline", "your_verdict"])
        for group, why, o in review:
            (REVIEW / group).mkdir(exist_ok=True)
            shutil.copy(REAL / "audio" / f"{o['clip']}.mp3", REVIEW / group / f"{o['clip']}.mp3")
            w.writerow([group, o["clip"], why, o["cer_finetuned"], o["seconds"], o["speech_seconds"],
                        o["expected_speech_s"], o["peak_dbfs"], o["snr_db"], o["Text"], o["finetuned"], o["baseline"], ""])
    for g in sorted({g for g, _, _ in review}):
        print(f"review {g:22} {sum(x[0] == g for x in review):3} clips")
    print(f"wrote {REVIEW}/review.csv and the clips in {REVIEW}/<group>/")


if __name__ == "__main__":
    main()
