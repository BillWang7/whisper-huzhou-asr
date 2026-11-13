# Data pipeline

How [BillWang7/huzhou-dialect-speech](https://huggingface.co/datasets/BillWang7/huzhou-dialect-speech) was
built: local Huzhou TV news broadcasts with burned-in Mandarin subtitles, turned into short
(audio clip, transcript) pairs by reading the subtitles with OCR. The reasoning behind each step is in
the [technical report, Section 2](../docs/technical-report.md#2-data); this page maps the steps to code
and shows how to run it.

## Steps

| Step | What happens | Where |
|---|---|---|
| 0. Manual | Download broadcasts, cut out Mandarin by hand | — |
| 1. Frames | Extract video frames | `build_dataset.py` |
| 2. OCR | Read the subtitle in each frame | `build_dataset.py`, `subtitles.frame_text` |
| 3. Segments | Group identical frames into timed subtitles | `subtitles.group_frames` |
| 4. Filter | Drop unreadable subtitles, strip punctuation | `subtitles.keep_confident` |
| 5. OCR flicker | Merge near-identical neighbours | `subtitles.merge_ocr_variants` |
| 6. Length | Drop very short subtitles | `subtitles.drop_short` |
| 7. Manual check | Remove clips found bad by listening | `bad_clips.json` |
| 8. Cut audio | Cut each subtitle's audio from the broadcast | `build_dataset.py` |
| 9. Deduplicate | Drop repeated transcripts | `subtitles.drop_duplicates` |
| 10. Split | Train/test split | `split.py` |

The thresholds are constants in the code; what they are and why is in the report.

## Run it

```bash
pip install -r data_pipeline/requirements.txt   # plus ffmpeg
python -m data_pipeline.build_dataset path/to/videos/ data/combined/
python -m data_pipeline.split data/combined/ data/
```

The output is `data/train/` and `data/test/`, each with `audio/<n>.mp3` and `text.csv`.
Videos are named by broadcast date (e.g. `20191202.mp4`), which is how `bad_clips.json` refers to them.
OCR is the slow step: a 40-minute broadcast is 36,000 frames, about 12 minutes at the ~50 frames/s the GPU managed. Its output is cached per video in
`pipeline_work/subtitles/`, so reruns are quick.

