"""Build (audio clip, transcript) pairs from news broadcasts with burned-in subtitles.

    python -m data_pipeline.build_dataset VIDEOS_DIR OUT_DIR [--work WORK_DIR] [--bad-clips FILE]

VIDEOS_DIR holds one .mp4 per broadcast, with Mandarin segments already cut out by hand. For each video:

1. ffmpeg extracts 15 frames per second.
2. PaddleOCR reads the subtitle text in every frame.
3. Frames become timed subtitle segments (see subtitles.py): runs of identical text are grouped,
   unreadable or low-confidence segments are dropped, punctuation is stripped, OCR flicker is merged,
   and segments of 1 second or less are dropped. Saved to WORK_DIR/subtitles/<video>.csv, so a rerun
   skips the slow OCR step.
4. Clips listed in --bad-clips (found by listening) are removed.
5. The audio of each remaining segment is cut out.

Finally clips from all videos are combined in video-name order, repeated transcripts are dropped (first
one kept), and the result is written as OUT_DIR/audio/<n>.mp3 and OUT_DIR/text.csv (column "Text").
Split it with `python -m data_pipeline.split`.

Needs ffmpeg and the packages in data_pipeline/requirements.txt. A GPU makes OCR much faster.
"""
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from natsort import natsorted
from pydub import AudioSegment

from data_pipeline.subtitles import drop_duplicates, frame_text, segment_ms, segments_from_frames

FPS = 15


def extract_frames(video, frames_dir):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"fps={FPS}",
                    str(frames_dir / "%d.png")], check=True)


def ocr_frames(ocr, frames_dir):
    """(text, confidence) for frames 1, 2, 3, ... in order."""
    frames = []
    for path in natsorted(frames_dir.glob("*.png"), key=lambda p: p.name):
        result = ocr.ocr(str(path), cls=True)[0] or []
        # each box is [corner points, (text, confidence)]; keep its left edge for ordering
        boxes = [(min(x for x, _ in points), text, confidence) for points, (text, confidence) in result]
        frames.append(frame_text(boxes))
    return frames


def subtitle_segments(video, work_dir, ocr):
    csv_path = work_dir / "subtitles" / f"{video.stem}.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            return [(int(r["Start Frame"]), int(r["End Frame"]), r["Text"], float(r["Confidence"]))
                    for r in csv.DictReader(f)]
    with tempfile.TemporaryDirectory() as frames_dir:
        extract_frames(video, Path(frames_dir))
        segments = segments_from_frames(ocr_frames(ocr, Path(frames_dir)), FPS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Frame", "End Frame", "Text", "Confidence"])
        writer.writerows(segments)
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("videos_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--work", type=Path, default=Path("pipeline_work"), help="intermediate files")
    parser.add_argument("--bad-clips", type=Path, default=Path(__file__).with_name("bad_clips.json"))
    args = parser.parse_args()

    from paddleocr import PaddleOCR  # imported here so the rest of the package works without it
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    bad_clips = json.loads(args.bad_clips.read_text(encoding="utf-8")) if args.bad_clips.exists() else {}

    texts, clips = [], []  # clips: (video path, start ms, end ms)
    for video in natsorted(args.videos_dir.glob("*.mp4"), key=lambda p: p.name):
        print(f"{video.name}: OCR", flush=True)
        segments = subtitle_segments(video, args.work, ocr)
        bad = set(bad_clips.get(video.stem, []))
        segments = [s for i, s in enumerate(segments) if i not in bad]
        print(f"{video.name}: {len(segments)} clips", flush=True)
        for segment in segments:
            texts.append(segment[2])
            clips.append((video, *segment_ms(segment, FPS)))

    keep = drop_duplicates(texts)
    print(f"{len(texts)} clips, {len(keep)} after removing repeated transcripts", flush=True)

    (args.out_dir / "audio").mkdir(parents=True, exist_ok=True)
    audio, loaded = None, None
    for n, i in enumerate(keep):
        video, start, end = clips[i]
        if video != loaded:
            # 44.1 kHz, the sample rate of the published clips
            audio, loaded = AudioSegment.from_file(video).set_frame_rate(44100), video
        audio[start:end].export(args.out_dir / "audio" / f"{n}.mp3", format="mp3")
    with open(args.out_dir / "text.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Text"])
        writer.writerows([texts[i]] for i in keep)
    print("done:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
