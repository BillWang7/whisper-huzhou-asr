"""API Lambda: the REST API behind the website (small and fast; no model here).

Routes (API Gateway HTTP API -> this function):
    GET  /                   the web page (same layout as the Gradio demo, app/demo.py)
    GET  /demo               clip lists for the two test sets and the featured clips
    GET  /audio/{set}/{file} a test clip's audio (set = news | real, file = N.mp3)
    POST /jobs               start a job: {"filename": ...} -> save a PENDING row, return a one-time S3 upload form;
                             {"example": {"set": ..., "clip": N}} -> copy that test clip to uploads/ (no form)
    GET  /jobs/{id}          one job's status and transcription; for a test clip, also the comparison with what
                             was said and with baseline Whisper (scored by src/scoring.py, like the Gradio demo)
    GET  /jobs               the 20 most recent finished jobs (history)

The browser uploads its own audio straight to S3 with the form from POST /jobs, so it never passes through this
function. Either way the file lands in uploads/<job_id>/, which triggers the transcriber Lambda.
The demo/ folder (test clips, transcripts, baseline outputs, a copy of src/scoring.py) is built by
aws/scripts/build_demo_data.py before deploying.
"""
import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB is several minutes of MP3; Whisper only hears 30 s anyway
UPLOAD_URL_SECONDS = 300
HISTORY_LIMIT = 20
PAGE = Path(__file__).with_name("index.html")
DEMO = Path(__file__).with_name("demo")
_demo = None


def demo_data() -> dict:
    global _demo
    if _demo is None:
        _demo = json.loads((DEMO / "data.json").read_text(encoding="utf-8"))
    return _demo


def _scoring():
    import importlib.util
    spec = importlib.util.spec_from_file_location("scoring", DEMO / "scoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table():
    import boto3
    return boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])


def _s3():
    import boto3
    from botocore.config import Config

    # Sign upload forms for the bucket's regional address (bucket.s3.us-east-2.amazonaws.com). The default global
    # address answers a new bucket outside us-east-1 with a 307 redirect for hours, and browsers won't follow
    # a redirect on a form upload.
    region = os.environ.get("AWS_REGION", "us-east-2")
    return boto3.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com",
                        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def respond(status: int, body, content_type: str = "application/json", headers: dict | None = None) -> dict:
    if content_type == "application/json":
        body = json.dumps(body, ensure_ascii=False, default=str)
    return {"statusCode": status, "headers": {"Content-Type": content_type, **(headers or {})}, "body": body}


def safe_filename(name: str) -> str:
    """Keep a readable file name but drop anything that could break an S3 key or a path."""
    name = Path(name or "audio").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")[:80]
    return name or "audio"


def public_job(item: dict) -> dict:
    keys = ["job_id", "status", "filename", "created_at", "finished_at", "text", "audio_seconds",
            "truncated", "inference_seconds", "total_seconds", "cold_start", "error"]
    return {k: item[k] for k in keys if k in item}


# ---------- Demo: test clips, like the Gradio app's first two tabs ----------

def example_clip(example) -> tuple[str, str] | None:
    """{"set": "news"|"real", "clip": N} -> ("news", "190") if it's a real test clip, else None."""
    if not isinstance(example, dict):
        return None
    name, clip = example.get("set"), str(example.get("clip", ""))
    if name in ("news", "real") and clip in demo_data()["sets"][name]["texts"]:
        return name, clip
    return None


def demo_index() -> dict:
    data = demo_data()
    clips = {name: sorted(int(c) for c in s["texts"]) for name, s in data["sets"].items()}
    samples = [{k: s[k] for k in ("label", "set", "clip", "english")} for s in data["samples"]]
    return respond(200, {"clips": clips, "samples": samples}, headers={"Cache-Control": "public, max-age=3600"})


def audio(name: str, file: str) -> dict:
    clip = file.removesuffix(".mp3")
    if not file.endswith(".mp3") or not example_clip({"set": name, "clip": clip}):
        return respond(404, {"error": "no such clip"})
    data = (DEMO / "audio" / name / file).read_bytes()
    return {"statusCode": 200, "isBase64Encoded": True, "body": base64.b64encode(data).decode(),
            "headers": {"Content-Type": "audio/mpeg", "Cache-Control": "public, max-age=86400"}}


def scored(scoring, reference: str, prediction: str) -> dict:
    """Error rate plus the character alignment the page draws. A runaway output (the model stuck repeating) is cut
    to 2x the reference + 10 characters for display, as in app/demo.py; the error rate still counts all of it."""
    cer, correct, length = scoring.score(reference, prediction)
    shown, limit = scoring.normalize(prediction), 2 * len(scoring.normalize(reference)) + 10
    return {"cer": round(cer, 1), "correct": correct, "length": length,
            "pairs": scoring.align(reference, shown[:limit]), "hidden": max(len(shown) - limit, 0)}


def comparison(name: str, clip: str, ours: str) -> dict:
    """The same comparison as app/demo.py's run_sample / run_test, as data for the page to draw."""
    scoring, data = _scoring(), demo_data()
    s = data["sets"][name]
    reference, baseline = s["texts"][clip], s["baseline"].get(clip)
    sample = next((x for x in data["samples"] if x["set"] == name and str(x["clip"]) == clip), None)
    result = {"set": name, "clip": int(clip), "reference": reference, "english": "", "caption": "",
              "ours": {**scored(scoring, reference, ours), "note": ""},
              "baseline": {**scored(scoring, reference, baseline), "note": ""} if baseline is not None else None}
    if sample:  # a featured clip: English meaning and the hand-written notes
        result["english"] = sample["english"]
        if ours == sample["expected_finetuned"] and sample["finetuned_note"]:
            result["ours"]["note"] = sample["finetuned_note"]
        elif result["ours"]["cer"] == 0:
            result["ours"]["note"] = "Exact match with what was said."
        if result["baseline"]:
            result["baseline"]["note"] = sample.get("baseline_note",
                                                    "Gibberish: it has nothing to do with what was said.")
    else:
        label = "News broadcasts" if name == "news" else "Real speakers"
        result["caption"] = f"{label} clip #{clip}. Reference transcript from examples/{s['folder']}/text.csv."
    return result


def create_example_job(name: str, clip: str) -> dict:
    """Transcribe a test clip: copy it into uploads/<job_id>/, which triggers the transcriber like any upload."""
    job_id = uuid.uuid4().hex
    filename = f"{name}_{clip}.mp3"
    _table().put_item(Item={
        "job_id": job_id, "kind": "job", "status": "PENDING", "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(), "example_set": name, "example_clip": clip,
    })
    _s3().put_object(Bucket=os.environ["BUCKET_NAME"], Key=f"uploads/{job_id}/{filename}",
                     Body=(DEMO / "audio" / name / f"{clip}.mp3").read_bytes(), ContentType="audio/mpeg")
    return respond(201, {"job_id": job_id})


def create_job(body: dict) -> dict:
    if "example" in body:
        found = example_clip(body["example"])
        if not found:
            return respond(400, {"error": "unknown test clip"})
        return create_example_job(*found)
    filename = safe_filename(body.get("filename", ""))
    job_id = uuid.uuid4().hex
    key = f"uploads/{job_id}/{filename}"
    _table().put_item(Item={
        "job_id": job_id,
        "kind": "job",  # every row shares this value so the ByTime index can sort all jobs by time
        "status": "PENDING",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    upload = _s3().generate_presigned_post(
        Bucket=os.environ["BUCKET_NAME"],
        Key=key,
        Conditions=[["content-length-range", 1, MAX_UPLOAD_BYTES]],
        ExpiresIn=UPLOAD_URL_SECONDS,
    )
    return respond(201, {"job_id": job_id, "upload": upload})


def get_job(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        return respond(400, {"error": "bad job id"})
    item = _table().get_item(Key={"job_id": job_id}).get("Item")
    if not item:
        return respond(404, {"error": "job not found"})
    job = public_job(item)
    if item.get("status") == "DONE" and item.get("example_set"):
        job["comparison"] = comparison(item["example_set"], str(item["example_clip"]), item.get("text", ""))
    return respond(200, job)


def list_jobs() -> dict:
    from boto3.dynamodb.conditions import Key

    resp = _table().query(
        IndexName="ByTime",
        KeyConditionExpression=Key("kind").eq("job"),
        ScanIndexForward=False,  # newest first
        Limit=HISTORY_LIMIT * 2,
    )
    done = [public_job(i) for i in resp.get("Items", []) if i.get("status") == "DONE"]
    return respond(200, {"jobs": done[:HISTORY_LIMIT]})


def handler(event, context=None):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    try:
        if method == "GET" and path == "/":
            return respond(200, PAGE.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if method == "POST" and path == "/jobs":
            body = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                import base64
                body = base64.b64decode(body).decode("utf-8")
            return create_job(json.loads(body))
        if method == "GET" and path == "/jobs":
            return list_jobs()
        if method == "GET" and path == "/demo":
            return demo_index()
        if method == "GET" and path.startswith("/audio/") and path.count("/") == 3:
            _, _, name, file = path.split("/")
            return audio(name, file)
        if method == "GET" and path.startswith("/jobs/"):
            return get_job(path.removeprefix("/jobs/"))
        return respond(404, {"error": "not found"})
    except json.JSONDecodeError:
        return respond(400, {"error": "body must be JSON"})
