"""Bundle what the website's demo tabs need into aws/api/demo/ (run before `sam build`; the folder is gitignored).

    python3 aws/scripts/build_demo_data.py          # from the repo root

Writes:
    aws/api/demo/scoring.py         a copy of src/scoring.py, so the site's error rates match the Gradio demo exactly
    aws/api/demo/data.json          both test sets' transcripts + precomputed baseline outputs, and the featured clips
    aws/api/demo/audio/{news,real}/N.mp3   every test clip (~130 MB; served by the API, and copied to uploads/ to
                                           transcribe, exactly like a user's upload)
Uses only the standard library.
"""
import csv
import filecmp
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
OUT = ROOT / "aws/api/demo"
SETS = {"news": "test_set", "real": "real_speakers"}  # same two held-out test sets as app/demo.py


def load_set(folder: str) -> dict:
    path = EXAMPLES / folder
    with open(path / "text.csv", encoding="utf-8") as f:
        texts = {int(r.get("clip", i)): r["Text"] for i, r in enumerate(csv.DictReader(f))}
    with open(path / "baseline.csv", encoding="utf-8") as f:
        baseline = {int(r["clip"]): r["baseline_prediction"] for r in csv.DictReader(f)}
    return {"folder": folder, "texts": texts, "baseline": baseline}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    shutil.copy(ROOT / "src/scoring.py", OUT / "scoring.py")

    sets = {}
    for name, folder in SETS.items():
        s = load_set(folder)
        dest = OUT / "audio" / name
        dest.mkdir()
        for clip in s["texts"]:
            shutil.copy(EXAMPLES / folder / "audio" / f"{clip}.mp3", dest / f"{clip}.mp3")
        sets[name] = {"folder": folder, "texts": {str(k): v for k, v in sorted(s["texts"].items())},
                      "baseline": {str(k): v for k, v in sorted(s["baseline"].items())}}

    # Featured clips (samples.json): examples/test_N.mp3 / real_N.mp3 are copies of test-set clip N
    samples = []
    for i, s in enumerate(json.loads((EXAMPLES / "samples.json").read_text(encoding="utf-8"))):
        prefix, clip = s["file"].removesuffix(".mp3").split("_")
        name = {"test": "news", "real": "real"}[prefix]
        same = filecmp.cmp(EXAMPLES / s["file"], OUT / "audio" / name / f"{clip}.mp3", shallow=False)
        assert same, f"{s['file']} differs from {SETS[name]}/audio/{clip}.mp3"
        assert sets[name]["texts"][clip] == s["reference"], f"{s['file']}: reference differs from text.csv"
        label = f"Clip {i + 1} ({'news' if name == 'news' else 'real speaker'})"
        samples.append({"label": label, "set": name, "clip": int(clip), **{k: s[k] for k in s if k != "file"}})

    (OUT / "data.json").write_text(json.dumps({"sets": sets, "samples": samples}, ensure_ascii=False),
                                   encoding="utf-8")
    n_audio = sum(1 for _ in (OUT / "audio").rglob("*.mp3"))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}: {n_audio} clips, {len(samples)} featured, {size:.0f} MB")


if __name__ == "__main__":
    main()
