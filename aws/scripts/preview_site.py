"""Preview the website locally, with AWS and the model faked out (nothing is sent anywhere, nothing costs money).

    python3 aws/scripts/build_demo_data.py      # once, builds aws/api/demo/
    python3 aws/scripts/preview_site.py         # then open http://localhost:8765/

Runs the real API Lambda code (aws/api/app.py) behind a tiny local web server. S3 and DynamoDB are replaced by
in-memory fakes, and the "transcriber" answers after 3 seconds with the fine-tuned model's saved output for test clips
(examples/*/finetuned.csv) or a placeholder for your own audio, so the page's layout and flow can be checked.
"""
import csv
import email.parser
import email.policy
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
spec = importlib.util.spec_from_file_location("api_app", ROOT / "aws/api/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

SAVED = {}
for name, folder in {"news": "test_set", "real": "real_speakers"}.items():
    with open(ROOT / "examples" / folder / "finetuned.csv", encoding="utf-8") as f:
        SAVED[name] = {r["clip"]: r["finetuned_prediction"] for r in csv.DictReader(f)}
JOBS = {}


class FakeTable:
    def put_item(self, Item):
        JOBS[Item["job_id"]] = dict(Item)

    def get_item(self, Key):
        item = JOBS.get(Key["job_id"])
        return {"Item": dict(item)} if item else {}


def fake_transcribe(job_id: str, delay: float = 3.0):
    def done():
        time.sleep(delay)
        job = JOBS[job_id]
        if job.get("example_set"):
            text = SAVED[job["example_set"]].get(job["example_clip"], "")
        else:
            text = "（本地预览：部署后这里显示模型的转写）"
        job.update(status="DONE", text=text, audio_seconds="3.2", truncated=False, inference_seconds="13.4",
                   total_seconds="14.1", cold_start=False)
    threading.Thread(target=done, daemon=True).start()


class FakeS3:
    def put_object(self, Key, **kwargs):  # an example clip copied to uploads/<job_id>/
        fake_transcribe(Key.split("/")[1])

    def generate_presigned_post(self, Key, **kwargs):  # the browser will POST its file to /_fake_s3
        return {"url": "/_fake_s3", "fields": {"key": Key}}


class Handler(BaseHTTPRequestHandler):
    def _lambda(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if method == "POST" and self.path == "/_fake_s3":  # the "S3 upload": read the key, start the fake job
            msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
            key = next(p.get_content() for p in msg.iter_parts() if p.get_param("name", header="content-disposition") == "key")
            fake_transcribe(key.split("/")[1])
            self.send_response(204)
            self.end_headers()
            return
        event = {"requestContext": {"http": {"method": method}}, "rawPath": self.path.split("?")[0],
                 "body": body.decode("utf-8") if body else None}
        r = app.handler(event)
        payload = r["body"]
        data = __import__("base64").b64decode(payload) if r.get("isBase64Encoded") else payload.encode("utf-8")
        self.send_response(r["statusCode"])
        for k, v in r["headers"].items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._lambda("GET")

    def do_POST(self):
        self._lambda("POST")

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")


if __name__ == "__main__":
    with mock.patch.object(app, "_table", return_value=FakeTable()), \
         mock.patch.object(app, "_s3", return_value=FakeS3()), \
         mock.patch.dict("os.environ", {"BUCKET_NAME": "local", "TABLE_NAME": "local"}):
        print(f"Preview at http://localhost:{PORT}/  (AWS and the model are faked; Ctrl-C to stop)")
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
