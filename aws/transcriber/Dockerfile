# Transcriber Lambda image. Build from the repo root so src/decoding.py can be copied in:
#   docker build --platform linux/arm64 -f aws/transcriber/Dockerfile -t whisper-lambda .
# AWS's official Lambda base image; it includes the runtime that talks to Lambda and a local emulator.
FROM public.ecr.aws/lambda/python:3.12

# CPU-only PyTorch (Lambda has no GPU)
COPY aws/transcriber/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the 8-bit model into the image so a cold start doesn't download anything (see quantize.py for why
# quantizing can't happen inside Lambda). Download, quantize and delete the 3 GB float32 weights in ONE layer,
# so they never end up in the image: it stays well under Lambda's 10 GB image limit. The tokenizer and config
# files stay in /opt/model.
ARG MODEL_ID=BillWang7/whisper-medium-ch-hz
ARG MODEL_REVISION=main
COPY aws/transcriber/quantize.py /tmp/quantize.py
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}', local_dir='/opt/model', \
allow_patterns=['*.json', '*.txt', 'model.safetensors'])" \
    && python /tmp/quantize.py /opt/model /opt/model-int8.pt \
    && rm -rf /opt/model/model.safetensors /opt/model/.cache \
    && ls -la /opt/model /opt/model-int8.pt

# Never reach out to the internet at run time; Lambda's file system is read-only except /tmp
ENV MODEL_DIR=/opt/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf

COPY src/decoding.py ${LAMBDA_TASK_ROOT}/src/decoding.py
COPY aws/transcriber/app.py ${LAMBDA_TASK_ROOT}/app.py

CMD ["app.handler"]
