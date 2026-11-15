# Serverless deployment on AWS

The fine-tuned model ([BillWang7/whisper-medium-ch-hz](https://huggingface.co/BillWang7/whisper-medium-ch-hz)) as a
public website that costs nothing while nobody is using it.

**Live:** https://huzhou-asr.com (also www.huzhou-asr.com, and the AWS-generated address https://jo3j4k9ieg.execute-api.us-east-2.amazonaws.com/; stack
`whisper-huzhou`, us-east-2 / Ohio)

**Status (15 Nov 2026): working.** Upload an audio file on the page; the transcript appears about 15–17 s later, or
about 25–60 s later if nobody has used the site for a while (a "cold start": Lambda has to start a fresh copy).

**Gradio-style page (deployed 15 Nov 2026):** the page now has the Gradio demo's
layout: header stats, and three tabs (featured clips with English meanings, random test clip with a slider, your own
audio), with the same scored comparison against what was said and against baseline Whisper. A "Why does it take
about 15 seconds?" box explains the serverless trade-off. See "The website" below.

```
Browser ──GET /──────────────▶ API Gateway ──▶ API Lambda ──▶ web page
Browser ──POST /jobs─────────▶ API Gateway ──▶ API Lambda ──▶ DynamoDB (job: PENDING)
                                                          └──▶ one-time S3 upload form
Browser ──upload audio───────▶ S3 (uploads/<job_id>/...)
                                 │ "new file" event
                                 ▼
                        Transcriber Lambda (container, x86_64, 3 GB: CTranslate2 int8 Whisper-medium)
                                 │
                                 ▼
                        DynamoDB (job: DONE, text, timings)
Browser ──GET /jobs/<id>─────▶ API Gateway ──▶ API Lambda ──▶ DynamoDB  (polls every 2 s)
Browser ──GET /jobs──────────▶ API Gateway ──▶ API Lambda ──▶ DynamoDB  (recent history)
```

| Folder / file | What it is |
|---|---|
| `template.yaml` | Every AWS resource, as code (AWS SAM / CloudFormation) |
| `samconfig.toml` | Saved `sam build` / `sam deploy` settings (stack name, Region, BuildKit on) |
| `transcriber/Dockerfile.ct2` | **The transcriber image in use.** Stage 1 converts the published model to CTranslate2 int8; stage 2 is the Lambda image (no PyTorch), ~1 GB |
| `transcriber/app.py` | S3 event → download → ffmpeg decode → Hugging Face spectrogram → CTranslate2 → DynamoDB (`ENGINE=ct2`). Also keeps the PyTorch path (`ENGINE=torch`) |
| `transcriber/requirements-ct2.txt`, `requirements-convert.txt` | Pinned packages for the Lambda image and for the conversion stage |
| `transcriber/Dockerfile`, `quantize.py`, `requirements.txt` | The earlier PyTorch int8 image. **Doesn't work on Lambda** (see below); kept for reference |
| `api/` | Small Python Lambda: the REST API and the web page (`index.html`, same layout as `app/demo.py`) |
| `api/demo/` | **Generated, gitignored.** Every test clip (2,524 MP3s, 127 MB), transcripts, baseline outputs, featured clips, and a copy of `src/scoring.py`. Built by `scripts/build_demo_data.py`; deployed inside the API Lambda |
| `scripts/build_demo_data.py` | Builds `api/demo/`. **Run before every `sam build`** |
| `scripts/preview_site.py` | The website on your Mac with AWS and the model faked: http://localhost:8765/ |
| `scripts/smoke_test.py` | End-to-end test of the live site, exactly as a browser does it |
| `scripts/eval_live.py` | Accuracy of the **deployed** site: uploads the first N test clips, compares with the published model |
| `scripts/measure_local.py` | Run an image in Lambda's local emulator at 3 GB / 2 CPUs: time per clip, peak memory |
| `scripts/eval_int8.py` | Accuracy of an image's model run locally in Docker (first N test clips) |
| `tests/` | Unit tests for both Lambdas, AWS and the model faked out (19 tests) |

## Results (measured 15 Nov 2026)

**Accuracy of the live site** (`scripts/eval_live.py`, first 200 news and first 100 real-speaker test clips uploaded
through the website, 0 errors):

| | Published model (PyTorch float32) | **Live site** (CTranslate2 int8, x86) |
|---|---:|---:|
| News, first 200 test clips | 23.8% CER | **24.6% CER** (138/200 outputs identical) |
| Real speakers, first 100 clips (all read-aloud) | 53.9% CER | **55.2% CER** |

**Speed and memory on Lambda** (x86_64, 3,008 MB ≈ 1.7 vCPUs; CloudWatch REPORT lines and `smoke_test.py`):

| | Time |
|---|---:|
| Warm request, model time | ~13–14 s per clip (median 13.3 s news, 14.0 s real speakers) |
| Warm request, upload to text on the page | ~15–17 s |
| Cold start, upload to text | 26 s (image cached) to 61 s (after ~20 min idle); the first request after a deploy can take 2–3 min |
| Max memory used | ~2.2 GB of 3.0 GB |

**Cost.** ~14 s × 3 GB ≈ 42 GB-seconds per clip, about $0.0007 at x86 prices. Idle: only storage (the ~1 GB image
in ECR, cents a month). The whole night of testing (≈ 330 transcriptions) cost well under $1 of the $140 credits.

## How we got the model onto Lambda (the full story)

**The constraint.** New AWS projects can give a Lambda function at most **3,008 MB** of memory. The normal maximum is
10,240 MB, but for new accounts AWS raises the cap on its own as you use Lambda, and it **can't be requested**
(AWS's troubleshooting docs: "You cannot request memory quota increases"; no memory quota appears in our Service
Quotas list). Whisper-medium in 32-bit floats needs about 4–5 GB while running, so it doesn't fit.

**Attempt 1, PyTorch int8: failed.** Dynamic int8 quantization (`quantize.py`) kept accuracy (23.9% / 54.5% on the
same clips) but on real Lambda PyTorch's QNNPACK engine fails with "failed to initialize QNNPACK" (it worked in the
local emulator), and the oneDNN engine needed 4.2 GB and over 300 s per clip.

**Attempt 2, faster-whisper int8 on arm64: worked, but slower and less accurate.** CTranslate2 is a separate inference
engine built to run Whisper on CPUs; faster-whisper is the usual wrapper around it. Results: news 25.1%, real 55.6%;
on Lambda arm64 ~19 s warm, 89 s cold.

**Diagnosis.** faster-whisper in float32 (no quantization) still scored 25.0% on news, so most of the news gap wasn't
int8 at all. The temperature fallback never fired. The difference was faster-whisper's own spectrogram code: feeding
CTranslate2 Hugging Face's spectrogram instead brought news to 24.5% and real speakers to 54.8% (locally, arm64).

**Final version (deployed).**

1. `Dockerfile.ct2` stage 1 (runs natively on the build machine): download the published model, pinned to commit
   `610b9056…`, convert with `ct2-transformers-converter --quantization int8` (3 GB → 742 MB), and keep Hugging
   Face's feature-extractor and tokenizer files.
2. Stage 2, the Lambda image: CTranslate2 + Hugging Face `transformers` (only its numpy feature extractor and
   tokenizer) + ffmpeg. No PyTorch.
3. `app.py` decodes like `src/decoding.py` and reads its settings from that file: Chinese, transcribe, greedy at
   temperature 0, and if the output looks like a repetition loop (Hugging Face's token compression ratio > 1.35), sample
   again at 0.2, 0.4 … 1.0.
4. **x86_64 instead of arm64**: measured on Lambda, warm ~13.5 s vs ~19 s, cold ~55 s vs ~89 s. x86 costs 25% more per
   second, so it's about the same per clip.

**Tried and rejected:** int16 instead of int8 on x86 used 2,999 of the 3,008 MB (would crash on longer audio) and was
slower (16 s warm).

**Gotchas found on the way** (all fixed, and noted in the code):

- `CT2_THREADS` must equal the number of vCPUs. In Docker on the Mac it saw 10 cores but had 2 and ran ~9× slower.
- Building an x86 image on an Apple-silicon Mac: SAM's default Docker builder can't mix platforms, so BuildKit is on
  (`use_buildkit = true` in `samconfig.toml`), and the conversion stage runs natively: under x86 emulation,
  CTranslate2's library fails to load ("cannot enable executable stack").
- faster-whisper imports `requests` without declaring it (no longer used, but worth knowing).

## The website

Same layout and wording as the Gradio demo (`app/demo.py`):

| Tab | What happens when you press Transcribe |
|---|---|
| 🎧 Featured clips | The API Lambda copies that test clip into `uploads/<job_id>/`, which triggers the transcriber exactly like an upload. When it's done, `GET /jobs/<id>` adds the comparison: English meaning, error-rate tiles for our model and baseline Whisper, and the character-by-character grids, scored by the same `src/scoring.py` as the Gradio demo |
| 🎲 Random test clip | Same, for any clip of either test set (slider or 🎲) |
| 🎙️ Your own audio | Upload or record; the browser uploads straight to S3 (as before); shows the text only |

Baseline Whisper's outputs are precomputed (as in the Gradio demo); only our model runs live. The header numbers are
the published full-precision results. The page plays clips through `GET /audio/<set>/<N>.mp3` (served from the API
Lambda's bundle, so the bucket stays private) and gets the clip lists from `GET /demo`.
Checked on the live site after deploying (15 Nov 2026): featured clip 1 in 26 s including a cold start (live 8-bit
output scored 13% there vs 7% for the full model: same 14 of 15 characters correct plus one extra), a random
real-speaker clip in 16 s warm, and an uploaded MP3 in 16 s warm. Checked locally with `scripts/preview_site.py`: all three tabs, scores identical to `src/scoring.py` (e.g. featured
clip 1: ours 7%, baseline 93%), phone width and dark mode. 19 unit tests (`tests/`) cover the new routes.

## Design choices

- **Audio goes straight from the browser to S3** with a presigned POST form (max 10 MB), so it never passes through
  API Gateway (10 MB / 30 s limits) or the API Lambda. Forms are signed for the bucket's regional address
  (`s3.us-east-2.amazonaws.com`): a new bucket outside us-east-1 answers the global address with a 307 redirect for a
  while, and browsers won't follow redirects on uploads.
- **The model is baked into the container image**, so a cold start doesn't download it. It's loaded on the first
  request rather than at import, because loading can take longer than Lambda's 10-second init phase.
- **Asynchronous jobs**: the S3 event invokes the transcriber asynchronously, and the page polls `GET /jobs/{id}`. A
  cold start can take longer than API Gateway's 30-second limit, so a synchronous request would time out.
- **Audio is cut at 30 s**, the same as the Gradio demo (Whisper hears 30 s at a time).
- **Own domain**: `huzhou-asr.com` is registered in Route 53 (auto-renew, privacy on). The template's optional
  `DomainName` / `HostedZoneId` settings (set in `samconfig.toml`) add a free ACM certificate, API Gateway custom
  domains for the bare name and `www`, and Route 53 alias records. Leave them empty to deploy without a domain.
- **Cost and abuse guards**: API rate limit (5 req/s, burst 10), 10 MB upload cap, private bucket, uploads deleted
  after 7 days, on-demand DynamoDB.

## Commands

From the repo root unless noted. SAM needs Docker Desktop's socket:
`export DOCKER_HOST="unix://$HOME/.docker/run/docker.sock"`. AWS credentials: `aws login --profile bill`.

```bash
# unit tests (no AWS needed)
cd aws && python3 -m unittest discover tests

# build the transcriber image for the Mac (arm64) and test it in Lambda's emulator at 3 GB / 2 CPUs
docker build --platform linux/arm64 -f aws/transcriber/Dockerfile.ct2 -t whisper-lambda-ct2 .
CT2_THREADS=2 python3 aws/scripts/measure_local.py whisper-lambda-ct2 examples/test_190.mp3 examples/real_566.mp3

# preview the website locally (AWS and the model faked)
python3 aws/scripts/build_demo_data.py && python3 aws/scripts/preview_site.py

# deploy (builds the x86 image with BuildKit; ~5 min when cached, ~15 min from scratch)
python3 aws/scripts/build_demo_data.py      # the test clips + data for the demo tabs (aws/api/demo/)
cd aws && sam build && sam deploy --profile bill

# test the live site, and measure its accuracy (first 200 news + 100 real-speaker clips, ~15 min, < $1)
python3 aws/scripts/smoke_test.py https://huzhou-asr.com/ examples/test_190.mp3
python3 aws/scripts/eval_live.py https://huzhou-asr.com/ 200 100

# delete everything from AWS
cd aws && sam delete --stack-name whisper-huzhou --profile bill
```

**To deploy a newer model** (e.g. after retraining): upload it to Hugging Face, set `MODEL_REVISION` in
`transcriber/Dockerfile.ct2` to the new commit, `sam build && sam deploy`, then run `eval_live.py`.
