"""Measure a transcriber image the way Lambda would run it: 3,008 MB memory, 2 CPUs, the Lambda emulator.

    python3 aws/scripts/measure_local.py whisper-lambda-ct2 examples/test_190.mp3 examples/real_675.mp3 ...

Starts the container (repo mounted at /repo), invokes it once per file through the emulator, and prints each
transcription, the time per call (the first includes loading the model, like a cold start) and the container's
peak memory. Standard library + docker only.
"""
import json
import subprocess
import sys
import time
import urllib.request

image, files = sys.argv[1], sys.argv[2:]
name = "whisper-measure"
subprocess.run(["docker", "rm", "-f", name], capture_output=True)
env_args = [x for k, v in subprocess.os.environ.items() if k.startswith(("CT2_", "TORCH_", "OMP_"))
            for x in ("-e", f"{k}={v}")]  # pass thread settings through, e.g. CT2_THREADS=2
subprocess.run(["docker", "run", "-d", "--name", name, *env_args, "--platform", "linux/arm64", "--memory", "3008m",
                "--memory-swap", "3008m", "--cpus", "2", "-p", "9123:8080", "-v", f"{subprocess.os.getcwd()}:/repo:ro",
                image], check=True, capture_output=True)
time.sleep(3)
url = "http://localhost:9123/2015-03-31/functions/function/invocations"
try:
    for path in files:
        t = time.perf_counter()
        req = urllib.request.Request(url, data=json.dumps({"audio_path": f"/repo/{path}"}).encode())
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        print(f"{path}: {time.perf_counter() - t:.1f}s  {json.dumps(out, ensure_ascii=False)}", flush=True)
    peak = subprocess.run(["docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak"],
                          capture_output=True, text=True).stdout.strip()
    print(f"peak memory: {int(peak) / 2**30:.2f} GiB" if peak else "peak memory: (not available)")
finally:
    logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    if "Error" in logs.stdout + logs.stderr:
        print((logs.stdout + logs.stderr)[-3000:])
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
