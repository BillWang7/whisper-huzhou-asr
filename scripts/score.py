"""Score both models on both test sets. Every result number in the README, report, demo and model card
comes from this script.

    python -m scripts.score

CER uses src/scoring.py, the same function the demo shows per clip (punctuation ignored). The 95%
confidence interval is from 2,000 bootstrap resamples of the clips.
"""
import csv
import random
import statistics
from pathlib import Path

from src.scoring import score

SETS = [
    ("News broadcasts", Path("examples/test_set")),
    ("Real speakers", Path("examples/real_speakers")),
]
MODELS = [("baseline", "baseline_prediction"), ("finetuned", "finetuned_prediction")]
LENGTHS = [("<=9 chars", 0, 9), ("10-14 chars", 10, 14), (">=15 chars", 15, 10**6)]  # reference length


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: int(r["clip"])) if rows and "clip" in rows[0] else rows


def ci95(values: list[float]) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return means[49], means[1949]


def line(label: str, cers: list[float]) -> str:
    low, high = ci95(cers)
    return (f"  {label:25} n={len(cers):5}  mean CER {statistics.mean(cers):5.1f}% (95% CI {low:.1f}-{high:.1f})  "
            f"median {statistics.median(cers):5.1f}%  exact {sum(c == 0 for c in cers):4}  over 100% {sum(c > 100 for c in cers):4}")


for set_name, folder in SETS:
    refs = read_rows(folder / "text.csv")
    print(set_name)
    for model, column in MODELS:
        if not (folder / f"{model}.csv").exists():
            print(f"  {model}: no results yet (python -m scripts.run_test_set {model} --set ...)")
            continue
        preds = [r[column] for r in read_rows(folder / f"{model}.csv")]
        assert len(preds) == len(refs), f"{folder}/{model}.csv: {len(preds)} rows, expected {len(refs)}"
        scored = [score(r["Text"], p) for r, p in zip(refs, preds)]
        cers = [s[0] for s in scored]
        print(line(model, cers))
        subsets = sorted({r["set"] for r in refs if "set" in r})
        for subset in subsets:
            print(line(f"  {model}, {subset}", [c for c, r in zip(cers, refs) if r["set"] == subset]))
        # Short references are penalised more: one wrong character in 5 is 20%, in 20 it is 5%.
        for label, low, high in LENGTHS:
            print(line(f"  {model}, {label}", [s[0] for s in scored if low <= s[2] <= high]))
