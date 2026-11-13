"""Turn per-frame OCR results into timed subtitle segments.

Pure Python (no OCR or audio libraries), so every rule can be unit-tested. The rules and thresholds are
the ones used to build the published dataset.

A segment is a tuple (start_frame, end_frame, text, confidence). Frames are numbered from 1, as ffmpeg
names them, and `confidence` is the lowest OCR confidence seen in the segment.
"""
import re

NO_TEXT = "No Text Detected"

# Chinese and ASCII punctuation removed from subtitles.
PUNCTUATION = "·！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
_PUNCTUATION_RE = re.compile("[%s]+" % re.escape(PUNCTUATION))


def frame_text(boxes, min_confidence=0.9):
    """Combine the text boxes OCR found in one frame into (text, confidence).

    `boxes` is a list of (left_x, text, confidence). A subtitle with a gap in it can come back as several
    boxes, so they are joined left to right. If any of several boxes is below `min_confidence`, the whole
    frame is treated as unreadable. A single box is returned as is; its confidence is checked later.
    """
    if not boxes:
        return NO_TEXT, 0.0
    if len(boxes) == 1:
        _, text, confidence = boxes[0]
        return text, confidence
    if any(confidence < min_confidence for _, _, confidence in boxes):
        return NO_TEXT, 0.0
    ordered = sorted(boxes, key=lambda box: box[0])
    return "".join(text for _, text, _ in ordered), min(confidence for _, _, confidence in ordered)


def group_frames(frames):
    """Merge runs of consecutive frames with identical text into segments.

    `frames` is the list of (text, confidence) for frames 1, 2, 3, ...
    """
    segments = []
    start = 1
    text, confidence = frames[0]
    for number, (frame_text_, frame_confidence) in enumerate(frames[1:], start=2):
        if frame_text_ == text:
            confidence = min(confidence, frame_confidence)
        else:
            segments.append((start, number - 1, text, confidence))
            start, text, confidence = number, frame_text_, frame_confidence
    segments.append((start, len(frames), text, confidence))
    return segments


def keep_confident(segments, min_confidence=0.9):
    """Drop unreadable or low-confidence segments and strip punctuation from the rest."""
    return [
        (start, end, _PUNCTUATION_RE.sub("", text), confidence)
        for start, end, text, confidence in segments
        if text != NO_TEXT and confidence > min_confidence
    ]


def levenshtein(a, b):
    """Edit distance between two strings."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def merge_ocr_variants(segments, max_distance=3):
    """Fix OCR flicker: merge neighbouring segments whose texts differ by at most `max_distance` edits.

    A brief misread splits one subtitle into two or three near-identical segments. The merged segment spans
    both, and keeps the text that was on screen longer (the misread is usually the shorter one).
    """
    segments = list(segments)
    i = 0
    while i < len(segments) - 1:
        start, end, text, confidence = segments[i]
        next_start, next_end, next_text, next_confidence = segments[i + 1]
        if levenshtein(text, next_text) <= max_distance:
            if end - start >= next_end - next_start:
                segments[i : i + 2] = [(start, next_end, text, confidence)]
            else:
                segments[i : i + 2] = [(start, next_end, next_text, next_confidence)]
        else:
            i += 1
    return segments


def drop_short(segments, fps, min_ms=1000):
    """Drop segments shown for `min_ms` or less (at 15 fps: 15 frames or fewer)."""
    min_frames = round(min_ms / (1000 / fps))
    return [s for s in segments if s[1] - s[0] + 1 > min_frames]


def segment_ms(segment, fps):
    """Audio (start, end) in milliseconds for a segment.

    Uses start_frame / fps, so a clip starts one frame (~67 ms) after the subtitle appeared.
    """
    start, end, _, _ = segment
    return start * 1000 / fps, end * 1000 / fps


def segments_from_frames(frames, fps):
    """The full subtitle pipeline for one video."""
    segments = keep_confident(group_frames(frames))
    segments = merge_ocr_variants(segments)
    return drop_short(segments, fps)


def drop_duplicates(texts):
    """Indices of the first occurrence of each transcript (later repeats are dropped)."""
    seen, keep = set(), []
    for i, text in enumerate(texts):
        if text not in seen:
            seen.add(text)
            keep.append(i)
    return keep
