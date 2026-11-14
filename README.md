# Huzhou Dialect Speech Recognition

Speech-to-text for **Huzhou dialect (湖州话)**, a Wu Chinese dialect that standard speech
recognition can't handle. I fine-tuned OpenAI's Whisper-medium on about 13 hours of Huzhou dialect
audio. On held-out news clips it cuts the character error rate from **98.8% to 26.7%**, and on
real people reading everyday sentences from **93.5% to 55.6%**.

This repo is an interactive demo with three tabs:

- **Featured clips:** hand-picked test clips with English translations, compared against baseline Whisper
- **Random test clip:** run any of the 2,533 held-out clips (news broadcasts or real speakers), or roll a random one
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

Character error rate (CER, lower is better) on two held-out test sets, neither used in training:

| Test set | Clips | Whisper-medium (baseline) | **Fine-tuned (this project)** |
|---|---:|---:|---:|
| News broadcasts (same channel as training) | 1,705 | 98.8% | **26.7%** |
| Real speakers (everyday sentences, mostly phone recordings) | 828 | 93.5% | **55.6%** |

Mean CER per clip; both models use identical decoding settings ([`src/decoding.py`](src/decoding.py)).
On news, the fine-tuned model is better on 1,678 clips and gets 278 exactly right. On real speakers it is
better on 723 clips: still far ahead of the baseline, but with twice the news error rate, because people
speak differently from TV anchors and the model learned news phrasing (it sometimes inserts news words like
女士, "Ms."). Full method, confidence intervals and error analysis:
**[technical report](docs/technical-report.md)**.

**Reading the 55.6%.** CER counts changed characters, not wrong sentences, so it overstates how much is lost:
- **Synonyms count as errors.** Huzhou dialect has no standard written form, so there is more than one correct
  way to write a sentence in Mandarin. The model often picks a different one: 他们未必等得了半年 → 他们不一定等得了半年
  ("they may not be able to wait half a year") means the same thing but scores 33%.
- **Short sentences are penalised more.** Errors are divided by the sentence length, so one swapped word in a
  6-character sentence costs 17–33%. 42% of real-speaker sentences have 9 characters or fewer (21% on news), and
  those score worst (58.9% vs 50.7% for 15+ characters).

How often the meaning is fully kept hasn't been measured; try the real-speaker clips in the demo to judge.

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
examples/                8 featured clips (with English translations)
examples/test_set/       News test set: 1,705 clips (audio/), transcripts (text.csv),
                         per-clip outputs of both models (baseline.csv, finetuned.csv)
examples/real_speakers/  Real-speaker test set: 828 clips, same layout
docs/technical-report.md Full write-up
Dockerfile               Reproducible environment (Python 3.10, CPU PyTorch)
requirements.txt         Exact package versions
```

## Reproduce the results

On a machine with Python and the packages from `requirements.txt` (a GPU makes this much faster):

```bash
python -m scripts.run_test_set baseline                 # news test set
python -m scripts.run_test_set finetuned
python -m scripts.run_test_set baseline --set real      # real speakers
python -m scripts.run_test_set finetuned --set real
python -m scripts.score                                 # prints every number above
```

## Known Limitations

- Trained only on news broadcast audio. It works on ordinary speakers too, but with about twice the errors
  (55.6% vs 26.7% CER).
- Training clips are less than 15 seconds long, so the model struggles with longer audio samples. 