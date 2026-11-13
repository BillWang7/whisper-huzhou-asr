"""Split the combined clips into train and test, as was done for the published dataset.

    python -m data_pipeline.split DATA_DIR OUT_DIR

The split is sequential, not random: clips are in broadcast order, the first 90% go to train and the rest
to test, so the test set comes from the last broadcasts in that order. Writes OUT_DIR/train and
OUT_DIR/test, each with audio/<n>.mp3 (numbered from 0) and text.csv.
"""
import argparse
import csv
import shutil
from pathlib import Path


def split_index(n_clips, train_fraction=0.9):
    """Number of clips that go to train.

    Clips 0..round(0.9 * n) inclusive go to train, one more than exactly 90%.
    For the 17,059 published clips this gives 15,354 train and 1,705 test.
    """
    return round(train_fraction * n_clips) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    with open(args.data_dir / "text.csv", encoding="utf-8") as f:
        texts = [row["Text"] for row in csv.DictReader(f)]
    n_train = split_index(len(texts))
    for name, indices in [("train", range(n_train)), ("test", range(n_train, len(texts)))]:
        (args.out_dir / name / "audio").mkdir(parents=True, exist_ok=True)
        with open(args.out_dir / name / "text.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Text"])
            for new, old in enumerate(indices):
                shutil.copy(args.data_dir / "audio" / f"{old}.mp3", args.out_dir / name / "audio" / f"{new}.mp3")
                writer.writerow([texts[old]])
        print(f"{name}: {len(indices)} clips")


if __name__ == "__main__":
    main()
