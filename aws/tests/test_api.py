"""Tests for the API Lambda, with AWS faked out. Run from aws/: python -m unittest discover tests"""
import json
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load(name, folder):
    """Both Lambdas have an app.py, so load each one under its own module name."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / folder / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _load("api_app", "api")

os.environ.update(TABLE_NAME="jobs", BUCKET_NAME="audio")


def event(method, path, body=None):
    e = {"requestContext": {"http": {"method": method}}, "rawPath": path}
    if body is not None:
        e["body"] = json.dumps(body)
    return e


class Api(unittest.TestCase):
    def setUp(self):
        self.table = mock.MagicMock()
        self.s3 = mock.MagicMock()
        self.s3.generate_presigned_post.return_value = {"url": "https://s3", "fields": {"key": "k"}}
        mock.patch.object(app, "_table", return_value=self.table).start()
        mock.patch.object(app, "_s3", return_value=self.s3).start()
        self.addCleanup(mock.patch.stopall)

    def test_page_is_html(self):
        r = app.handler(event("GET", "/"))
        self.assertEqual(r["statusCode"], 200)
        self.assertIn("text/html", r["headers"]["Content-Type"])
        self.assertIn("<title>", r["body"])

    def test_create_job_saves_pending_row_and_returns_upload_form(self):
        r = app.handler(event("POST", "/jobs", {"filename": "my clip.mp3"}))
        self.assertEqual(r["statusCode"], 201)
        body = json.loads(r["body"])
        item = self.table.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["job_id"], body["job_id"])
        self.assertEqual(item["status"], "PENDING")
        self.assertEqual(item["filename"], "my_clip.mp3")
        post = self.s3.generate_presigned_post.call_args.kwargs
        self.assertEqual(post["Key"], f"uploads/{body['job_id']}/my_clip.mp3")
        self.assertIn(["content-length-range", 1, app.MAX_UPLOAD_BYTES], post["Conditions"])

    def test_filenames_cannot_escape_the_upload_folder(self):
        self.assertEqual(app.safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(app.safe_filename("录音 1.m4a"), "1.m4a")
        self.assertEqual(app.safe_filename(""), "audio")

    def test_bad_json_is_400(self):
        e = event("POST", "/jobs")
        e["body"] = "{not json"
        self.assertEqual(app.handler(e)["statusCode"], 400)

    def test_get_job(self):
        self.table.get_item.return_value = {"Item": {"job_id": "a" * 32, "status": "DONE", "text": "侬好", "kind": "job"}}
        r = app.handler(event("GET", "/jobs/" + "a" * 32))
        self.assertEqual(r["statusCode"], 200)
        body = json.loads(r["body"])
        self.assertEqual(body["text"], "侬好")
        self.assertNotIn("kind", body)  # internal fields stay internal

    def test_unknown_or_malformed_job_ids(self):
        self.table.get_item.return_value = {}
        self.assertEqual(app.handler(event("GET", "/jobs/" + "b" * 32))["statusCode"], 404)
        self.assertEqual(app.handler(event("GET", "/jobs/../secret"))["statusCode"], 400)

    def test_history_only_lists_finished_jobs_newest_first(self):
        self.table.query.return_value = {"Items": [
            {"job_id": "1", "status": "DONE", "text": "新"},
            {"job_id": "2", "status": "PENDING"},
            {"job_id": "3", "status": "DONE", "text": "旧"},
        ]}
        with mock.patch.dict(sys.modules, {"boto3.dynamodb.conditions": mock.MagicMock()}):
            r = app.handler(event("GET", "/jobs"))
        self.assertEqual([j["job_id"] for j in json.loads(r["body"])["jobs"]], ["1", "3"])
        self.assertFalse(self.table.query.call_args.kwargs["ScanIndexForward"])

    def test_unknown_route_is_404(self):
        self.assertEqual(app.handler(event("DELETE", "/jobs"))["statusCode"], 404)


class Demo(unittest.TestCase):
    """The Gradio-style demo routes, with a tiny made-up demo/ folder."""

    def setUp(self):
        import shutil
        import tempfile
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        (self.dir / "audio/news").mkdir(parents=True)
        (self.dir / "audio/news/190.mp3").write_bytes(b"ID3fake")
        shutil.copy(Path(__file__).resolve().parents[2] / "src/scoring.py", self.dir / "scoring.py")
        data = {"sets": {
                    "news": {"folder": "test_set", "texts": {"190": "来湖州旅游", "5": "记者来到"},
                             "baseline": {"190": "到5G的利益", "5": "这里"}},
                    "real": {"folder": "real_speakers", "texts": {"39": "你好"}, "baseline": {}}},
                "samples": [{"label": "Clip 1 (news)", "set": "news", "clip": 190, "reference": "来湖州旅游",
                             "english": "Visiting Huzhou", "expected_finetuned": "到湖州旅游",
                             "finetuned_note": "Small slip.", "baseline": "到5G的利益"}]}
        (self.dir / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        mock.patch.object(app, "DEMO", self.dir).start()
        mock.patch.object(app, "_demo", None).start()
        self.table, self.s3 = mock.MagicMock(), mock.MagicMock()
        mock.patch.object(app, "_table", return_value=self.table).start()
        mock.patch.object(app, "_s3", return_value=self.s3).start()
        self.addCleanup(mock.patch.stopall)

    def test_demo_index_lists_clips_and_featured(self):
        body = json.loads(app.handler(event("GET", "/demo"))["body"])
        self.assertEqual(body["clips"], {"news": [5, 190], "real": [39]})
        self.assertEqual(body["samples"][0]["english"], "Visiting Huzhou")
        self.assertNotIn("expected_finetuned", body["samples"][0])  # answers stay on the server

    def test_audio_route_serves_only_known_clips(self):
        r = app.handler(event("GET", "/audio/news/190.mp3"))
        self.assertEqual(r["headers"]["Content-Type"], "audio/mpeg")
        self.assertTrue(r["isBase64Encoded"])
        for bad in ("/audio/news/191.mp3", "/audio/news/..%2Fdata.json", "/audio/secret/190.mp3", "/audio/news/190"):
            self.assertEqual(app.handler(event("GET", bad))["statusCode"], 404, bad)

    def test_example_job_copies_the_clip_into_uploads(self):
        r = app.handler(event("POST", "/jobs", {"example": {"set": "news", "clip": 190}}))
        self.assertEqual(r["statusCode"], 201)
        job_id = json.loads(r["body"])["job_id"]
        item = self.table.put_item.call_args.kwargs["Item"]
        self.assertEqual((item["example_set"], item["example_clip"], item["status"]), ("news", "190", "PENDING"))
        put = self.s3.put_object.call_args.kwargs
        self.assertEqual(put["Key"], f"uploads/{job_id}/news_190.mp3")  # the transcriber's trigger prefix
        self.assertEqual(put["Body"], b"ID3fake")
        self.s3.generate_presigned_post.assert_not_called()

    def test_unknown_example_is_400(self):
        for ex in ({"set": "news", "clip": 999}, {"set": "../x", "clip": 1}, "news/190"):
            self.assertEqual(app.handler(event("POST", "/jobs", {"example": ex}))["statusCode"], 400)

    def test_finished_featured_job_has_scored_comparison(self):
        self.table.get_item.return_value = {"Item": {"job_id": "a" * 32, "status": "DONE", "text": "到湖州旅游",
                                                     "example_set": "news", "example_clip": "190"}}
        c = json.loads(app.handler(event("GET", "/jobs/" + "a" * 32))["body"])["comparison"]
        self.assertEqual(c["english"], "Visiting Huzhou")
        self.assertEqual((c["ours"]["cer"], c["ours"]["correct"], c["ours"]["length"]), (20.0, 4, 5))
        self.assertEqual(c["ours"]["note"], "Small slip.")  # output matches the expected one
        self.assertEqual(c["ours"]["pairs"][0], ["来", "到"])
        self.assertTrue(c["baseline"]["note"].startswith("Gibberish"))

    def test_random_test_clip_has_caption_and_exact_match_note_only_for_featured(self):
        self.table.get_item.return_value = {"Item": {"job_id": "a" * 32, "status": "DONE", "text": "记者来到",
                                                     "example_set": "news", "example_clip": "5"}}
        c = json.loads(app.handler(event("GET", "/jobs/" + "a" * 32))["body"])["comparison"]
        self.assertEqual(c["ours"]["cer"], 0)
        self.assertEqual(c["ours"]["note"], "")
        self.assertIn("clip #5", c["caption"])

    def test_runaway_output_is_cut_for_display_but_fully_scored(self):
        self.table.get_item.return_value = {"Item": {"job_id": "a" * 32, "status": "DONE", "text": "你" * 100,
                                                     "example_set": "real", "example_clip": "39"}}
        c = json.loads(app.handler(event("GET", "/jobs/" + "a" * 32))["body"])["comparison"]
        self.assertEqual(c["ours"]["hidden"], 100 - (2 * 2 + 10))
        self.assertEqual(c["ours"]["cer"], 4950.0)  # 1 wrong + 98 extra, over 2 reference characters
        self.assertIsNone(c["baseline"])  # no precomputed baseline for this clip


if __name__ == "__main__":
    unittest.main()
