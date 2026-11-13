# Huzhou Dialect Speech Recognition

Speech-to-text for **Huzhou dialect (湖州话)**, a Wu Chinese dialect that standard speech
recognition can't handle. I fine-tuned OpenAI's Whisper-medium on about 15 hours of Huzhou dialect
audio, and cut the character error rate from **120.7% to 22.8%**.

This repo is an interactive demo with three tabs:

- **Featured clips:** hand-picked test clips with English translations, compared against baseline Whisper
- **Random test clip:** run any of the 1,705 held-out test clips yourself, or roll a random one
- **Your own audio:** upload a file or record yourself

| | |
|---|---|
| **Model** | [BillWang7/whisper-medium-ch-hz](https://huggingface.co/BillWang7/whisper-medium-ch-hz) |
| **Dataset** | [BillWang7/huzhou-dialect-speech](https://huggingface.co/datasets/BillWang7/huzhou-dialect-speech) |

## Why

My grandmother mainly speaks Huzhou dialect. When I looked for speech recognition that could understand
her, nothing worked: Whisper and other public models focus on Mandarin and Cantonese, and output
gibberish on regional dialects that millions of people still speak. So I built one.

## Results

Character error rate (CER) on held-out test clips the model never trained on:

| Model | CER |
|---|---|
| Whisper-medium (baseline) | 120.7% |
| **Fine-tuned (this project)** | **22.8%** |

Measured on the same 100 test clips. On the full 1,705-clip test set, the fine-tuned model scores 26.8%.

| | Example |
|---|---|
| What was said | 青塘小区是一个高度老龄化的老小区 *("Qingtang is an old neighborhood with a large elderly population")* |
| Fine-tuned | 青塘小区是一个高度老龄化的老小区 ✅ |
| Baseline | 青銅小池是一個高的龍領湖的龍小池 ❌ *(gibberish: "bronze small pond is a high dragon...")* |

## Run the demo

You need [Docker](https://www.docker.com/products/docker-desktop/). No GPU required.

```bash
git clone https://github.com/BillWang7/whisper-huzhou-asr.git
cd whisper-huzhou-asr
docker build -t whisper-huzhou-asr .
docker run --rm -p 7860:7860 -v whisper-hf-cache:/root/.cache/huggingface whisper-huzhou-asr
```

Then open **http://localhost:7860**.

- The first run downloads the model (~3 GB), so it takes a few minutes. Later runs start in under a minute.
- It runs on CPU, so each transcription takes several seconds.
- Stop it with `Ctrl+C`.

## How it works

1. **Data:** audio extracted from Huzhou dialect news broadcasts and film, with transcripts made by
   reading the on-screen subtitles with OCR and aligning them to the audio, then cleaned by hand.
   Result: 15,354 training clips and 1,705 test clips.
2. **Training:** fine-tuned `openai/whisper-medium` for 4,000 steps (batch size 64, learning rate 1e-5).
3. **Demo:** a [Gradio](https://www.gradio.app/) web app, packaged with Docker so it runs the same on any machine.

Training code and per-clip results are in the [model repo](https://huggingface.co/BillWang7/whisper-medium-ch-hz/tree/main).

## Project layout

```
app/demo.py          Gradio web app (featured clips, random test clip, your own audio)
src/transcribe.py    Loads the model and transcribes audio
examples/            5 featured clips (with English translations) + precomputed baseline outputs
examples/test_set/   Full held-out test set: 1,705 clips, transcripts (text.csv),
                     baseline outputs for clips 0–99 (baseline.csv)
Dockerfile           Reproducible environment (Python 3.10, CPU PyTorch)
requirements.txt     Exact package versions
```

## Limitations

- Trained mostly on news audio, so it's less accurate on casual speech and noisy recordings.
- Names and place names are the most common errors.
- Writes Mandarin characters rather than a phonetic transcription of the dialect.
