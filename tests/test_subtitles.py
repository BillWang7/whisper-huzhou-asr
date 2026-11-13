"""Tests for the dataset pipeline's subtitle rules. Run: python -m unittest discover tests"""
import unittest

from data_pipeline.split import split_index
from data_pipeline.subtitles import (
    NO_TEXT, drop_duplicates, drop_short, frame_text, group_frames, keep_confident, levenshtein,
    merge_ocr_variants, segment_ms, segments_from_frames,
)


class FrameText(unittest.TestCase):
    def test_no_boxes_is_unreadable(self):
        self.assertEqual(frame_text([]), (NO_TEXT, 0.0))

    def test_boxes_joined_left_to_right(self):
        boxes = [(500, "下午", 0.95), (100, "今天", 0.99)]
        self.assertEqual(frame_text(boxes), ("今天下午", 0.95))

    def test_one_low_confidence_box_makes_frame_unreadable(self):
        self.assertEqual(frame_text([(100, "今天", 0.99), (500, "下午", 0.5)]), (NO_TEXT, 0.0))

    def test_single_box_returned_as_is(self):
        self.assertEqual(frame_text([(100, "今天", 0.5)]), ("今天", 0.5))


class Segments(unittest.TestCase):
    def test_group_identical_runs(self):
        frames = [("a", 0.99), ("a", 0.95), ("b", 0.97), ("b", 0.98), ("b", 0.96), ("a", 0.99)]
        self.assertEqual(group_frames(frames), [(1, 2, "a", 0.95), (3, 5, "b", 0.96), (6, 6, "a", 0.99)])

    def test_keep_confident_strips_punctuation(self):
        segments = [(1, 20, "你好，世界。", 0.95), (21, 40, NO_TEXT, 0.0), (41, 60, "太低", 0.9)]
        self.assertEqual(keep_confident(segments), [(1, 20, "你好世界", 0.95)])

    def test_levenshtein(self):
        self.assertEqual(levenshtein("湖州新闻", "湖州新闻"), 0)
        self.assertEqual(levenshtein("湖州新闻", "潮州新闻网"), 2)
        self.assertEqual(levenshtein("", "abc"), 3)

    def test_ocr_flicker_merged_keeping_longer_text(self):
        segments = [(1, 30, "今天天气很好", 0.99), (31, 33, "今天夭气很好", 0.95), (34, 60, "明年见面吧朋友们", 0.98)]
        self.assertEqual(merge_ocr_variants(segments),
                         [(1, 33, "今天天气很好", 0.99), (34, 60, "明年见面吧朋友们", 0.98)])

    def test_flicker_keeps_next_text_when_it_lasted_longer(self):
        segments = [(1, 3, "今天夭气很好", 0.95), (4, 40, "今天天气很好", 0.99)]
        self.assertEqual(merge_ocr_variants(segments), [(1, 40, "今天天气很好", 0.99)])

    def test_chained_flicker_merges_into_one(self):
        segments = [(1, 30, "abcdef", 0.99), (31, 32, "abcxef", 0.95), (33, 34, "abcxyf", 0.95)]
        self.assertEqual(merge_ocr_variants(segments), [(1, 34, "abcdef", 0.99)])

    def test_drop_one_second_or_less(self):
        segments = [(1, 15, "15 frames", 0.99), (16, 31, "16 frames", 0.99)]
        self.assertEqual(drop_short(segments, fps=15), [(16, 31, "16 frames", 0.99)])

    def test_segment_ms(self):
        self.assertEqual(segment_ms((15, 30, "x", 0.99), fps=15), (1000.0, 2000.0))

    def test_full_pipeline(self):
        frames = [(NO_TEXT, 0.0)] * 5 + [("湖州新闻，", 0.99)] * 20 + [("湖州新闽，", 0.95)] * 2 + [("短", 0.99)] * 10
        self.assertEqual(segments_from_frames(frames, fps=15), [(6, 27, "湖州新闻", 0.99)])


class Dataset(unittest.TestCase):
    def test_drop_duplicates_keeps_first(self):
        self.assertEqual(drop_duplicates(["a", "b", "a", "c", "b"]), [0, 1, 3])

    def test_split_matches_published_sizes(self):
        n_train = split_index(17_059)
        self.assertEqual((n_train, 17_059 - n_train), (15_354, 1_705))


if __name__ == "__main__":
    unittest.main()
