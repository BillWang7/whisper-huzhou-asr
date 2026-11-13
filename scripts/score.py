"""Score baseline.csv and finetuned.csv against the reference transcripts.

    python -m scripts.score

CER uses src/scoring.py, the same function the demo shows per clip (punctuation ignored).
"""
import csv
import statistics
from pathlib import Path

from src.scoring import score

TEST_DIR = Path("examples/test_set")


def read(name: str, column: str) -> list[str]:
    with open(TEST_DIR / name, encoding="utf-8") as f:
        return [row[column] for row in sorted(csv.DictReader(f), key=lambda r: int(r.get("clip", 0)))]


refs = read("text.csv", "Text")
for name, column in [("baseline.csv", "baseline_prediction"), ("finetuned.csv", "finetuned_prediction")]:
    preds = read(name, column)
    assert len(preds) == len(refs), f"{name}: {len(preds)} rows, expected {len(refs)}"
    cers = [score(r, p)[0] for r, p in zip(refs, preds)]
    print(f"{name:15} mean CER {statistics.mean(cers):6.1f}%   median {statistics.median(cers):6.1f}%   "
          f"perfect clips {sum(c == 0 for c in cers):4}   clips over 100% {sum(c > 100 for c in cers):4}")
