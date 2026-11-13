
# Same Python version as the local `whisper` conda env (3.10.18).
# Debian 12 (bookworm) ships ffmpeg 5, which torchaudio 2.5.1 supports.
FROM python:3.10.18-slim-bookworm

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (exact versions from the local env).
# The extra index provides CPU-only PyTorch builds (avoids ~2.5 GB of NVIDIA
# CUDA libraries), so the image runs on any machine, with or without a GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY src/ ./src/
COPY app/ ./app/

# Configure Gradio to listen outside the container
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "-m", "app.demo"]
