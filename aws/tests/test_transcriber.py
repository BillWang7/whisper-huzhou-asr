"""Tests for the transcriber Lambda's plumbing, with the model and AWS faked out.
Run from aws/: python -m unittest discover tests"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


def _load(name, folder):
    """Both Lambdas have an app.py, so load each one under its own module name."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / folder / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transcriber = _load("transcriber_app", "transcriber")

os.environ.setdefault("TABLE_NAME", "jobs")
JOB = "c" * 32


def s3_event(key):
    return {"Records": [{"s3": {"bucket": {"name": "audio"}, "object": {"key": key}}}]}


class Transcriber(unittest.TestCase):
    def setUp(self):
        self.table = mock.MagicMock()
        self.s3 = mock.MagicMock()
        mock.patch.object(transcriber, "_table", return_value=self.table).start()
        mock.patch.object(transcriber, "_s3", return_value=self.s3).start()
        self.addCleanup(mock.patch.stopall)

    def saved(self):
        kw = self.table.update_item.call_args.kwargs
        return {name: kw["ExpressionAttributeValues"][":" + name] for name in
                (n[1:] for n in kw["ExpressionAttributeNames"])}

    def test_job_id_from_key(self):
        self.assertEqual(transcriber.job_id_from_key(f"uploads/{JOB}/clip.mp3"), JOB)
        self.assertEqual(transcriber.job_id_from_key(f"uploads/{JOB}/my+clip.mp3"), JOB)
        with self.assertRaises(ValueError):
            transcriber.job_id_from_key("other/clip.mp3")

    def test_success_saves_text_and_timings(self):
        result = {"text": "侬好", "audio_seconds": 3.2, "truncated": False, "inference_seconds": 6.8}
        with mock.patch.object(transcriber, "transcribe", return_value=result):
            transcriber.handler(s3_event(f"uploads/{JOB}/clip.mp3"))
        self.assertEqual(self.table.update_item.call_args.kwargs["Key"], {"job_id": JOB})
        row = self.saved()
        self.assertEqual(row["status"], "DONE")
        self.assertEqual(row["text"], "侬好")
        self.assertEqual(row["inference_seconds"], "6.8")  # DynamoDB can't take Python floats
        self.assertIn("finished_at", row)

    def test_failure_is_recorded_so_the_page_stops_waiting(self):
        with mock.patch.object(transcriber, "transcribe", side_effect=ValueError("audio is empty")):
            with self.assertRaises(ValueError):
                transcriber.handler(s3_event(f"uploads/{JOB}/clip.mp3"))
        row = self.saved()
        self.assertEqual(row["status"], "ERROR")
        self.assertIn("audio is empty", row["error"])

    def test_ct2_path_retries_at_next_temperature_when_output_loops(self):
        import sys
        import types

        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for src/decoding.py
        sys.modules.setdefault("ctranslate2", types.SimpleNamespace(
            set_random_seed=lambda seed: None, StorageView=types.SimpleNamespace(from_array=lambda a: a)))
        loop = [100] * 60        # the same token 60 times: compresses very well, so it counts as a loop
        normal = list(range(100, 130))
        model = mock.MagicMock()
        model.generate.side_effect = [[types.SimpleNamespace(sequences_ids=[ids + [50257]])] for ids in (loop, normal)]
        processor = mock.MagicMock(return_value=types.SimpleNamespace(input_features=np.zeros((1, 80, 3000))))
        processor.tokenizer.eos_token_id = 50257
        processor.tokenizer.decode.side_effect = lambda ids, skip_special_tokens: f"{len(ids)} tokens"
        with mock.patch.object(transcriber, "ENGINE", "ct2"), \
                mock.patch.object(transcriber, "get_model", return_value=(model, processor)), \
                mock.patch.object(transcriber, "load_audio", return_value=np.zeros(16000 * 5, np.float32)):
            out = transcriber.transcribe("clip.mp3")
        first, second = (c.kwargs for c in model.generate.call_args_list)
        self.assertEqual(first["beam_size"], 1)                    # greedy first, like src/decoding.py
        self.assertNotIn("sampling_temperature", first)
        self.assertEqual(second["sampling_temperature"], 0.2)      # then the next temperature
        self.assertEqual(out["text"], "30 tokens")                 # the non-looping attempt, end-of-text removed
        self.assertGreater(transcriber.hf_compression_ratio(loop), 1.35)
        self.assertLess(transcriber.hf_compression_ratio(normal), 1.35)


if __name__ == "__main__":
    unittest.main()
