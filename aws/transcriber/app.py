"""Transcriber Lambda: S3 upload -> Whisper -> DynamoDB.

Runs as a container image (the model is too big for a normal Lambda zip). AWS calls `handler` when a
file lands under `uploads/` in the audio bucket. It downloads the file, transcribes it with the
fine-tuned model, and writes the result to the jobs table so the website can show it.

Local smoke test (no AWS needed): invoke with {"audio_path": "/path/inside/container.mp3"}.
"""
import json
import os
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone

SAMPLE_RATE = 16000
MAX_SECONDS = 30  # Whisper hears 30 s at a time; longer audio is cut, the same as the Gradio demo
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
INT8_PATH = os.environ.get("INT8_PATH", "/opt/model-int8.pt")
ENGINE = os.environ.get("ENGINE", "torch")  # "ct2" = CTranslate2 int8 (Dockerfile.ct2), no PyTorch
CT2_MODEL_DIR = os.environ.get("CT2_MODEL_DIR", "/opt/model-ct2")  # converted model
PROCESSOR_DIR = os.environ.get("PROCESSOR_DIR", "/opt/processor")  # Hugging Face feature extractor + tokenizer files

_model = None
_processor = None


def job_id_from_key(key: str) -> str:
    """uploads/<job_id>/<filename> -> <job_id>"""
    parts = urllib.parse.unquote_plus(key).split("/")
    if len(parts) < 3 or parts[0] != "uploads" or not parts[1]:
        raise ValueError(f"unexpected upload key: {key}")
    return parts[1]


def load_audio(path: str):
    """Decode any audio format (mp3, wav, m4a, webm from a browser mic...) to 16 kHz mono float32."""
    import imageio_ffmpeg
    import numpy as np

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", path,
           "-t", str(MAX_SECONDS + 1), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def get_model():
    """Load the model on first use and keep it for later calls while this Lambda stays warm.

    Loaded lazily (not at import) because loading 3 GB takes longer than Lambda's 10-second init limit.
    """
    global _model, _processor
    if _model is None and ENGINE == "ct2":
        import ctranslate2
        from transformers import WhisperProcessor

        # One thread per vCPU: with more threads than cores it ran ~9x slower in testing
        _model = ctranslate2.models.Whisper(CT2_MODEL_DIR, device="cpu",
                                            compute_type=os.environ.get("CT2_COMPUTE_TYPE", "int8"),
                                            intra_threads=int(os.environ.get("CT2_THREADS") or os.cpu_count() or 1))
        _processor = WhisperProcessor.from_pretrained(PROCESSOR_DIR)
    if _model is None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor

        # One thread per vCPU; more threads than cores makes PyTorch slower, not faster
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS") or os.cpu_count() or 1))
        _processor = WhisperProcessor.from_pretrained(MODEL_DIR)
        if os.environ.get("QUANTIZE_INT8") == "1":
            # 8-bit copy made at build time by quantize.py: ~1 GB instead of 3 GB, much faster on CPU
            torch.backends.quantized.engine = os.environ.get("QUANT_ENGINE", "onednn")  # must match quantize.py
            _model = torch.load(INT8_PATH, weights_only=False).eval()
        else:
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_DIR).eval()
    return _model, _processor


def transcribe(path: str) -> dict:
    model, processor = get_model()
    audio = load_audio(path)
    audio_seconds = len(audio) / SAMPLE_RATE
    if audio_seconds < 0.1:
        raise ValueError("audio is empty or too short")
    start = time.perf_counter()
    if ENGINE == "ct2":
        text = _transcribe_ct2(model, processor, audio)
    else:
        text = _transcribe_torch(model, processor, audio)
    return {
        "text": text,
        "audio_seconds": round(min(audio_seconds, MAX_SECONDS), 2),
        "truncated": audio_seconds > MAX_SECONDS,
        "inference_seconds": round(time.perf_counter() - start, 2),
    }


def hf_compression_ratio(ids: list[int]) -> float:
    """Hugging Face's repetition measure (transformers' _retrieve_compression_ratio): token ids as 2-byte strings
    (vocabulary 51,865 < 65,536), zlib-compressed. A looping output compresses well, so its ratio is high."""
    import zlib

    raw = b"".join(int(t).to_bytes(2, "little") for t in ids)
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def _transcribe_ct2(model, processor, audio) -> str:
    """The same decoding as src/decoding.py (Hugging Face generate), done with CTranslate2.

    Hugging Face's own feature extractor makes the spectrogram: faster-whisper's version differs slightly and cost
    ~1 CER point on news in testing. Greedy at temperature 0; if the output looks like a loop (token compression
    ratio above the threshold), sample again at the next temperature; if every attempt loops, keep the last.
    """
    import ctranslate2
    from src.decoding import GENERATE_KWARGS, SEED

    features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features  # cuts at 30 s
    features = ctranslate2.StorageView.from_array(features)
    prompt = processor.tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", f"<|{GENERATE_KWARGS['language']}|>", f"<|{GENERATE_KWARGS['task']}|>",
         "<|notimestamps|>"])
    eos = processor.tokenizer.eos_token_id
    ctranslate2.set_random_seed(SEED)
    for temperature in GENERATE_KWARGS["temperature"]:
        options = dict(beam_size=1, max_length=448, suppress_blank=True, suppress_tokens=[-1])  # [-1]: model's list
        if temperature > 0:
            options.update(sampling_temperature=temperature, sampling_topk=0)  # sample from the full distribution
        ids = model.generate(features, [prompt], **options)[0].sequences_ids[0]
        text_ids = [t for t in ids if t < eos]  # special tokens (prompt, end of text) are >= eos
        if hf_compression_ratio(text_ids) <= GENERATE_KWARGS["compression_ratio_threshold"]:
            break
    return processor.tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def _transcribe_torch(model, processor, audio) -> str:
    import torch
    from src.decoding import generate  # same decoding settings as the demo and the evaluation

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        ids = generate(model, inputs.input_features, inputs.attention_mask)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def _table():
    import boto3
    return boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])


def _s3():
    import boto3
    return boto3.client("s3")


def save_result(job_id: str, fields: dict) -> None:
    """Update the job row. Floats are stored as strings because DynamoDB rejects Python floats."""
    fields = {k: (str(v) if isinstance(v, float) else v) for k, v in fields.items()}
    fields["finished_at"] = datetime.now(timezone.utc).isoformat()
    names = {f"#{k}": k for k in fields}
    values = {f":{k}": v for k, v in fields.items()}
    _table().update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET " + ", ".join(f"#{k} = :{k}" for k in fields),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def handle_upload(bucket: str, key: str) -> None:
    job_id = job_id_from_key(key)
    cold = _model is None
    local = f"/tmp/{job_id}"
    started = time.perf_counter()
    try:
        _s3().download_file(bucket, urllib.parse.unquote_plus(key), local)
        result = transcribe(local)
        result["total_seconds"] = round(time.perf_counter() - started, 2)
        result["cold_start"] = cold
        save_result(job_id, {"status": "DONE", **result})
        print(json.dumps({"job_id": job_id, **result}, ensure_ascii=False))
    except Exception as e:  # record the failure so the website stops waiting
        save_result(job_id, {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:500]})
        raise
    finally:
        if os.path.exists(local):
            os.remove(local)


def handler(event, context=None):
    if "audio_path" in event:  # local smoke test
        return transcribe(event["audio_path"])
    for record in event.get("Records", []):
        handle_upload(record["s3"]["bucket"]["name"], record["s3"]["object"]["key"])
    return {"ok": True}
