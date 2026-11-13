# Huzhou Dialect Speech Recognition

Speech-to-text for **Huzhou dialect (湖州话)**, a Wu Chinese dialect that standard speech
recognition can't handle. I fine-tuned OpenAI's Whisper-medium on about 13 hours of Huzhou dialect
audio, and cut the character error rate from **98.8% to 26.7%**.

This repo is an interactive demo with three tabs:

- **Featured clips:** hand-picked test clips with English translations, compared against baseline Whisper
- **Random test clip:** run any of the 1,705 held-out test clips yourself, or roll a random one
- **Your own audio:** upload a file or record yourself

| | |
|---|---|
| **Model** | [BillWang7/whisper-medium-ch-hz](https://huggingface.co/BillWang7/whisper-medium-ch-hz) |
| **Dataset** | [BillWang7/huzhou-dialect-speech](https://huggingface.co/datasets/BillWang7/huzhou-dialect-speech) |

## Inspiration

My grandmother mainly speaks Huzhou dialect. When I looked for speech recognition that could understand
her, nothing worked: Whisper and other public models focus on Mandarin and Cantonese, and output
gibberish on regional dialects that millions of people still speak. So I built one.

Huzhou dialect has about 3 million native speakers, but decades of Mandarin-only schooling mean
many younger people in the region can understand it but no longer speak it. Like most Chinese dialects, it
has no standard written form, so turning speech into simplified Chinese characters is one way to help preserve it.

## Results

Character error rate (CER) on held-out test clips the model never trained on:

| Model | CER |
|---|---|
| Whisper-medium (baseline) | 98.8% |
| **Fine-tuned (this project)** | **26.7%** |

Mean CER per-clip on all 1,705 test clips; both models use identical decoding settings
([`src/decoding.py`](src/decoding.py)). The fine-tuned model is better on 1,678 clips and gets 278 exactly
right; the baseline's median CER is 100%, meaning a typical clip gets essentially nothing right.
Full method, confidence intervals and error analysis: **[technical report](docs/technical-report.md)**.

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

1. **Data:** 72 Huzhou news broadcasts with Mandarin subtitles. Mandarin segments were removed by hand,
   subtitles were read with OCR, corrected and aligned to the audio: 15,354 training clips (12.9 h) and
   1,705 test clips (1.5 h). Code: [`data_pipeline/`](data_pipeline).
2. **Training:** full fine-tune of `openai/whisper-medium`, 4,000 steps, batch size 64, learning rate 1e-5,
   following Hugging Face's [Fine-Tune Whisper](https://huggingface.co/blog/fine-tune-whisper) guide.  Training notebooks are in the [model repo](https://huggingface.co/BillWang7/whisper-medium-ch-hz/tree/main).
3. **Demo:** a [Gradio](https://www.gradio.app/) web app, packaged with Docker so it runs the same on any machine.

The **[technical report](docs/technical-report.md)** covers the data pipeline, training curves, evaluation


## Project layout

```
app/demo.py              Gradio web app (featured clips, random test clip, your own audio)
src/transcribe.py        Loads the model and transcribes audio
src/decoding.py          Decoding settings shared by the demo and evaluation (incl. loop guard)
src/scoring.py           Character error rate, shared by the demo and evaluation
scripts/run_test_set.py  Runs either model over the whole test set (uses an Apple or NVIDIA GPU)
scripts/score.py         Scores both models' outputs
data_pipeline/           How the dataset was built: subtitle OCR, alignment, cleaning, train/test split
tests/                   Unit tests for the pipeline rules and CER (python -m unittest discover tests)
examples/                5 featured clips (with English translations)
examples/test_set/       Held-out test set: 1,705 clips (audio/), transcripts (text.csv),
                         per-clip outputs of both models (baseline.csv, finetuned.csv)
docs/technical-report.md Full write-up
Dockerfile               Reproducible environment (Python 3.10, CPU PyTorch)
requirements.txt         Exact package versions
```

## Reproduce the results

On a machine with Python and the packages from `requirements.txt` (a GPU makes this much faster):

```bash
python -m scripts.run_test_set baseline
python -m scripts.run_test_set finetuned
python -m scripts.score
```

## Known Limitations

- Trained only on news broadcast audio.
- Training clips are less than 15 seconds long, so the mdoel struggles with longer audio samples. 