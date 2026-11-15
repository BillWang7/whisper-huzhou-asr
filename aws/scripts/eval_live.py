"""Accuracy of the DEPLOYED site: upload the first N news and real-speaker test clips through the live API (exactly as
the web page does) and compare the transcripts with the published model's (examples/*/finetuned.csv).

    python3 aws/scripts/eval_live.py https://<api-id>.execute-api.us-east-2.amazonaws.com/ 200 100 [--workers 6]

Uses the real Lambda, so it measures what visitors get (CPU type and int8 numerics included). Runs several uploads at
once; each concurrent upload may start its own Lambda copy (cold start ~1 min). Cost at 3 GB: well under $1 for 300
clips. Writes aws/scripts/eval_live_<timestamp>.csv. Run from the repo root.
"""
import argparse
import csv
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for src.scoring
sys.path.insert(0, str(Path(__file__).resolve().parent))      # smoke_test.run
from smoke_test import run  # noqa: E402

from src.scoring import score  # noqa: E402

SETS = [("news", "examples/test_set"), ("real", "examples/real_speakers")]


def rows(path):
    with open(path, encoding="utf-8") as f:
        return {int(r.get("clip", n)): r for n, r in enumerate(csv.DictReader(f))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("n_news", type=int)
    p.add_argument("n_real", type=int)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    base = args.url.rstrip("/") + "/"
    jobs = []
    for (name, folder), n in zip(SETS, (args.n_news, args.n_real)):
        text, pub = rows(f"{folder}/text.csv"), rows(f"{folder}/finetuned.csv")
        jobs += [(name, folder, c, text[c]["Text"], pub[c]["finetuned_prediction"]) for c in sorted(text)[:n]]

    def one(job):
        name, folder, clip, ref, old = job
        for attempt in range(3):
            try:
                r = run(base, f"{folder}/audio/{clip}.mp3")
                return job, r
            except Exception as e:  # throttling or a network hiccup: wait and retry
                err = e
                time.sleep(5 * (attempt + 1))
        return job, {"status": "ERROR", "error": str(err)}

    out = Path(__file__).with_name(f"eval_live_{time.strftime('%Y%m%d_%H%M')}.csv")
    results = []
    with ThreadPoolExecutor(args.workers) as pool, open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "clip", "reference", "published", "live", "cer_published", "cer_live", "inference_s",
                    "cold_start"])
        for (name, folder, clip, ref, old), r in pool.map(one, jobs):
            new = r.get("text") or ""
            results.append((name, score(ref, old)[0], score(ref, new)[0], old == new, r))
            w.writerow([name, clip, ref, old, new, round(results[-1][1], 1), round(results[-1][2], 1),
                        r.get("inference_seconds"), r.get("cold_start")])
            f.flush()
    for name, _ in SETS:
        sub = [x for x in results if x[0] == name]
        if not sub:
            continue
        warm = [float(x[4]["inference_seconds"]) for x in sub
                if x[4].get("inference_seconds") and str(x[4].get("cold_start")).lower() == "false"]
        print(f"{name}: n={len(sub)}  published {statistics.mean(x[1] for x in sub):.1f}%  "
              f"live {statistics.mean(x[2] for x in sub):.1f}%  identical {sum(x[3] for x in sub)}/{len(sub)}  "
              f"errors {sum(x[4].get('status') != 'DONE' for x in sub)}  "
              f"warm inference {statistics.median(warm) if warm else float('nan'):.1f}s median")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
