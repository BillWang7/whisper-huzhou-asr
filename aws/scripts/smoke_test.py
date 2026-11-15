"""End-to-end test against the deployed site, doing exactly what the web page does.

Usage: python3 aws/scripts/smoke_test.py https://<api-id>.execute-api.<region>.amazonaws.com/ examples/test_190.mp3 [...]
Prints the transcription and timings for each file. Standard library + curl only.
"""
import json
import subprocess
import sys
import time
import urllib.request


def api(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run(base, path):
    started = time.perf_counter()
    job = api("POST", base + "jobs", {"filename": path.split("/")[-1]})
    up = job["upload"]
    cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", up["url"]]
    for k, v in up["fields"].items():
        cmd += ["--form-string", f"{k}={v}"]
    cmd += ["-F", f"file=@{path}"]
    code = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    if code != "204":
        raise SystemExit(f"upload failed: HTTP {code}")
    while True:
        time.sleep(1)
        j = api("GET", base + "jobs/" + job["job_id"])
        if j["status"] in ("DONE", "ERROR"):
            break
        if time.perf_counter() - started > 300:
            raise SystemExit("timed out")
    j["end_to_end_seconds"] = round(time.perf_counter() - started, 1)
    return j


if __name__ == "__main__":
    base = sys.argv[1].rstrip("/") + "/"
    for f in sys.argv[2:]:
        r = run(base, f)
        print(json.dumps({k: r.get(k) for k in ("status", "text", "error", "cold_start", "inference_seconds",
                                                 "total_seconds", "end_to_end_seconds")}, ensure_ascii=False))
