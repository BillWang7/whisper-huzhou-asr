# Preserving Huzhou Dialect by Fine-Tuning Whisper for Speech-to-Text

**Bill Wang** · Technical report · [Model](https://huggingface.co/BillWang7/whisper-medium-ch-hz) ·
[Dataset](https://huggingface.co/datasets/BillWang7/huzhou-dialect-speech) · [Code and demo](../README.md)

## Abstract

Huzhou dialect (湖州话) is a Wu Chinese dialect with roughly 3 million native speakers and no standard
written form. General-purpose speech recognition does not handle it: OpenAI's Whisper-medium reaches a
character error rate (CER) of 98.8% on our test set. We built a 14.4-hour supervised corpus from local news
broadcasts, using OCR on burned-in subtitles to obtain transcripts, and fine-tuned Whisper-medium on
12.9 hours of it. On 1,705 held-out news clips the fine-tuned model reaches **26.7% CER** (95% CI
25.5–27.8%), is better than the baseline on 1,678 clips, and transcribes 278 clips exactly. On a second test
set of 828 recordings of ordinary people reading everyday sentences, it reaches **55.6% CER** against the
baseline's 93.5%: a large gain, but twice the in-domain error rate.

## 1. Motivation

Chinese dialects carry local history and culture, but they are disappearing. For decades Mandarin has
been the language of schooling and most media, so in many places younger generations understand their
local dialect but no longer speak it. Most dialects are also only spoken, with no standard written form,
which makes them hard to record, search or study.

Speech-to-text is one way to help. Public models, Whisper included, are trained mostly on Mandarin and
Cantonese and produce unrelated text on smaller dialects (Section 5). This project focuses on one dialect,
Huzhou dialect from Huzhou, Zhejiang, which my family speaks, and asks how far a modest amount of labelled
audio can take an existing multilingual model.

The output is written in standard (simplified) Mandarin characters, the same convention used by the
subtitles we learned from. The task is therefore closer to dialect-to-Mandarin transcription than to a
phonetic transcription of the dialect.

## 2. Data

### 2.1 Source

Labelled Huzhou dialect audio is scarce. Supervised fine-tuning needs audio paired with text, so we used
a local Huzhou television news channel that broadcasts partly in Huzhou dialect and shows Mandarin
subtitles on screen. The broadcasts also contain Mandarin, mainly in interviews, which had to be removed.

### 2.2 Pipeline

The code is in [`data_pipeline/`](../data_pipeline).

1. **Collection and clipping.** 72 broadcast videos of about 40 minutes each were downloaded. Segments
   in Mandarin were removed by hand so that only dialect speech remained.
2. **Subtitle extraction.** Frames were sampled at 15 per second and read with PaddleOCR (Chinese
   model). When a subtitle came back as several text boxes, they were joined left to right; if any box
   had confidence below 0.9, the frame was treated as unreadable.
3. **Segmentation.** Consecutive frames with identical text were merged into one subtitle segment, whose
   first and last frames give its start and end time. Unreadable segments and segments with confidence
   of 0.9 or less were dropped, and punctuation was removed.
4. **OCR correction.** A momentary misread splits one subtitle into near-identical neighbours. Adjacent
   segments whose texts differ by at most 3 edits (Levenshtein distance) were merged, keeping the text
   that stayed on screen longer.
5. **Filtering and alignment.** Segments shown for 1 second or less were dropped. The audio for each
   remaining segment was cut from the broadcast using its frame times. Clips found to be wrong on
   listening were removed by hand (110 clips).
6. **Deduplication.** Clips were combined in broadcast order, and only the first clip of each exact
   transcript was kept.

### 2.3 Train/test split

The split is sequential, not random: with clips in broadcast order, the first 90% (15,354) went to
training and the remaining 1,705 to test. The test set is therefore drawn from the last broadcasts in
that order, and test clips  come from broadcasts that contributed nothing to training. This is a stricter test than a random clip-level split. The same news anchors, however, appear throughout the channel, so the
test set does not measure performance on unseen speakers.

### 2.4 Corpus statistics

| | Train | Test |
|---|---:|---:|
| Clips | 15,354 | 1,705 |
| Audio | 12.9 h | 1.5 h |
| Clip duration, mean (range) | 3.0 s (1.0–9.9 s) | 3.1 s (1.0–8.5 s) |
| Transcript length, mean (range) | 12.6 chars (1–32) | 13.3 chars (4–25) |

Audio is stored as 44.1 kHz stereo MP3 and converted to 16 kHz mono at load time. Every transcript is
unique within its split, and no transcript appears in both splits (exact-match check). The corpus is
published as [BillWang7/huzhou-dialect-speech](https://huggingface.co/datasets/BillWang7/huzhou-dialect-speech).

## 3. Model and training

We fine-tuned all weights of `openai/whisper-medium` (769M parameters), following Hugging Face's
[Fine-Tune Whisper](https://huggingface.co/blog/fine-tune-whisper) recipe, with Seq2SeqTrainer:

| Setting | Value |
|---|---|
| Steps | 4,000 (≈16.7 epochs) |
| Batch size | 64 (no gradient accumulation) |
| Learning rate | 1e-5, 500 warm-up steps, then linear decay; AdamW |
| Precision / memory | fp16, gradient checkpointing |
| Decoder prompt | language `zh`, task `transcribe` |
| Evaluation | every 1,000 steps on the test split, CER, keep best checkpoint |
| Training time | 3.3 hours |

Training curve (from the TensorBoard logs in the model repository; CER here is pooled over the test split
and counts punctuation, so it differs slightly from Section 5):

| Step | Train loss | Test loss | Test CER |
|---:|---:|---:|---:|
| 1,000 | 0.10 | 0.70 | 29.7% |
| 2,000 | 0.006 | 0.78 | 29.4% |
| 3,000 | 0.001 | 0.82 | 26.1% |
| 4,000 | 0.001 | 0.84 | 26.1% |

Training loss falls to about 0.001 by step 3,000 (≈12.5 epochs), so the model has essentially memorised
the training set, while test loss rises from step 1,000 onwards. This suggests that we may have overfitted the model. However, test CER still improves until step 3,000. Loss and CER can move in opposite directions because the loss is cross-entropy, the negative log of the probability given to the correct token. Once the model is near-certain of every answer, its remaining mistakes are confident ones, which the loss penalises heavily but CER counts only once.

What this implies:

- **The reported CER is still a fair held-out score.** Overfitting widens the gap between training and
  test performance, but no test clip was trained on, so 26.7% measures unseen clips. One caveat: the
  checkpoint was chosen by its test CER, which makes the score slightly optimistic; checkpoint scores
  ranged only from 26.1% to 29.7%, which bounds the effect.
- **Generalisation beyond this dataset is weaker.** Memorising the training set also means absorbing
  its particulars: the same anchors' voices, studio audio and recurring news phrasing. All audio is from
  one channel and the same anchors appear in both splits (Section 2.3), so accuracy on other speakers,
  conversation or noisy recordings is untested and likely lower.
- **Data, not training time, is the bottleneck.** Steps 3,000 and 4,000 score the same, so more training on
  this corpus would not help. Gains are more likely from more and more varied audio, and from
  regularisation: early stopping on a separate validation split, SpecAugment, label smoothing, or
  updating fewer weights (for example freezing the encoder, or LoRA).

## 4. Evaluation method

**Metric.** CER = (substitutions + deletions + insertions) / reference length, computed per clip after
removing punctuation and spaces from both strings (only letters, digits and Chinese characters are
compared). We report the mean over clips. CER can exceed 100% when the output is longer than the
reference. The implementation is [`src/scoring.py`](../src/scoring.py), the same function the demo uses.

**Decoding.** Both models are decoded with identical settings ([`src/decoding.py`](../src/decoding.py)):
language `zh`, task `transcribe`, and Whisper's temperature fallback. Whisper Baseline sometimes enters a
repetition loop and repeats a character until it hits the length limit; without a guard, one clip with an
8-character reference produced hundreds of repeated characters (CER 5,550%). With the guard, it changed
15 baseline outputs; the worst baseline clip fell from 5,550% to 333% CER.

**Test sets.** Two held-out sets, neither used in training. *News broadcasts* is the 1,705-clip test split
(Section 2.3), from the same channel as the training data. *Real speakers* (828 clips, 86.6 minutes) tests
the model outside that domain. It has two parts, both recorded with the speakers' permission: 728 clips
recorded by volunteers through a phone recording app, reading one sentence each, mostly everyday dialogue
from TV dramas and narration from travel programmes (8 kHz audio); and 100 clips from four longer recordings
reading a folk tale, a health leaflet and an essay (48 kHz), cut into sentences by hand. One app clip whose
sentence also appears in the training transcripts was removed; no other sentence overlaps the training data.

**Baseline.** The unmodified `openai/whisper-medium` with the same decoding settings. Forcing the language
matters: without it, Whisper's language detection produced Japanese, Arabic, French and other languages
on 9 of the first 100 clips.

## 5. Results

### 5.1 News broadcasts

| Model | Mean CER (95% CI) | Median CER | Exact clips | Clips over 100% |
|---|---:|---:|---:|---:|
| Whisper-medium (baseline) | 98.8% (97.8–99.8) | 100.0% | 1 | 505 |
| **Fine-tuned** | **26.7% (25.5–27.8)** | **21.7%** | **278** | 8 |

Confidence intervals are from 2,000 bootstrap resamples of the test clips. The fine-tuned model has lower
CER than the baseline on 1,678 clips, ties on 18 and is worse on 9. Pooled CER (total errors over total
reference characters) is 26.0% and 97.8%.

**Error types (fine-tuned).** 3,659 substitutions, 1,096 deletions and 1,136 insertions. Most errors are
wrong characters rather than missing or extra ones.

**By utterance length.**

| Reference length | Clips | Fine-tuned | Baseline |
|---|---:|---:|---:|
| ≤ 9 characters | 355 | 30.8% | 104.5% |
| 10–14 characters | 681 | 26.6% | 98.5% |
| ≥ 15 characters | 669 | 24.6% | 96.1% |

Short clips are hardest, as each error weighs more and there is less context.

**Examples** (held-out clips; these are the demo's featured clips):

| | Transcript | CER |
|---|---|---:|
| Reference | 青塘小区是一个高度老龄化的老小区 *(Qingtang is an old neighbourhood with many elderly residents)* | |
| Fine-tuned | 青塘小区是一个高度老龄化的老小区 | 0% |
| Baseline | 青銅小池是一個高的龍領湖的龍小池 | 56% |
| Reference | 这位老人叫席中科今年94岁了 *(This elderly man is Xi Zhongke, 94 this year)* | |
| Fine-tuned | 这位老人叫薛中科今年94岁 | 14% |
| Baseline | 这个老人家叫中科,跟你讲就是手术 | 71% |

Typical fine-tuned errors are personal and place names (席 → 薛 above) and near-synonyms (来 → 到).

### 5.2 Real speakers

| Model | Mean CER (95% CI) | Median CER | Exact clips | Clips over 100% |
|---|---:|---:|---:|---:|
| Whisper-medium (baseline) | 93.5% (92.2–94.9) | 100.0% | 0 | 114 |
| **Fine-tuned** | **55.6% (53.7–57.5)** | **50.5%** | **13** | 28 |

| Part | Clips | Fine-tuned | Baseline |
|---|---:|---:|---:|
| App recordings (everyday sentences, phone audio) | 728 | 55.8% | 94.6% |
| Read-aloud recordings (folk tale, leaflet, essay) | 100 | 53.9% | 85.5% |

The fine-tuned model is better than the baseline on 723 of 828 clips, ties on 54 and is worse on 51, and
it is better across all lengths (≤ 9 characters: 58.9% vs 97.1%; ≥ 15 characters: 50.7% vs 87.1%). Its
error rate is roughly double the news result, for three visible reasons:

- **Domain shift.** The model learned news language and sometimes imposes it: it inserted 女士 ("Ms.", as
  in news interviews' "Ms. X said") into 14 transcripts whose reference never contains it.
- **Paraphrase.** Training transcripts are Mandarin subtitles, which often reword the dialect. The model
  learned to do the same, so meaning-preserving rewrites count as errors: 他们未必等得了半年 → 他们不一定等得了半年
  (both "they may not be able to wait half a year") scores 33%. Errors are mostly substitutions (4,076,
  against 918 deletions and 539 insertions).
- **Recording conditions.** Most clips are 8 kHz phone recordings by untrained speakers, while training
  audio is studio-quality broadcast sound.
- **Short sentences.** CER divides the errors by the length of the reference, so on a short sentence each
  wrong character costs more: one synonym swap in a 6-character sentence is 17–33%, the same swap in a
  20-character sentence 5–10%. Real-speaker sentences are short (42% have 9 characters or fewer, against
  21% of news clips), and the short ones score worst (58.9%, against 50.7% for 15 characters or more).
  Length explains part of the gap, not all of it: even on long sentences the news error rate is half
  (24.6%).

**What 55.6% means in practice.** CER is not a percentage of sentences that are wrong or unintelligible. A
transcript at 50% CER often reads as the right sentence in different words, because Huzhou dialect has no
standard written form and there is more than one correct way to write it in Mandarin; the paraphrase
example above is a typical case. We have not measured how often the meaning is fully kept, which would need
human judgement; the featured and random clips in the demo let a reader check this directly.

CER therefore overstates how much meaning is lost, but the gap to the news result is real and is the main
target for future work: training data from more speakers and settings.


## 6. Reproducibility

```bash
# Transcribe each test set with each model (≈35 min per model for news, ≈20 for real speakers, Apple M4 GPU)
python -m scripts.run_test_set baseline
python -m scripts.run_test_set finetuned
python -m scripts.run_test_set baseline --set real
python -m scripts.run_test_set finetuned --set real
# Score everything (prints every result number in this report)
python -m scripts.score
```

To rebuild the dataset, see [`data_pipeline/`](../data_pipeline). The training code is `training.ipynb` in
the [model repository](https://huggingface.co/BillWang7/whisper-medium-ch-hz/tree/main); it calls the
splits `aqi_train` and `aqi_test`.

## 7. Data use

The audio comes from publicly available broadcasts, and copyright remains with the original broadcaster.
The dataset is shared for non-commercial research and education only, and content will be removed on
request.
