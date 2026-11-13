"""Tests for the character error rate. Run: python -m unittest discover tests"""
import unittest

from src.scoring import align, normalize, score


class CER(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(score("湖州新闻", "湖州新闻"), (0.0, 4, 4))

    def test_punctuation_and_spaces_ignored(self):
        self.assertEqual(normalize("你好， 世界！"), "你好世界")
        self.assertEqual(score("你好，世界。", "你好 世界")[0], 0.0)

    def test_one_substitution(self):
        self.assertEqual(score("这位老人叫席中科", "这位老人叫薛中科"), (12.5, 7, 8))

    def test_empty_prediction_is_100_percent(self):
        self.assertEqual(score("湖州", "")[0], 100.0)

    def test_extra_characters_can_exceed_100_percent(self):
        self.assertEqual(score("好", "好好好")[0], 200.0)

    def test_alignment_marks_insertions_and_deletions(self):
        self.assertEqual(align("abc", "abxc"), [("a", "a"), ("b", "b"), (None, "x"), ("c", "c")])
        self.assertEqual(align("abc", "ac"), [("a", "a"), ("b", None), ("c", "c")])


if __name__ == "__main__":
    unittest.main()
