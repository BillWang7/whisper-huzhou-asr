
import argparse

import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    AutoModelForSpeechSeq2Seq,
)

MODEL_ID = "BillWang7/whisper-medium-ch-hz"
SAMPLE_RATE = 16000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-medium"
)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID
).to(DEVICE)

model.eval()


def preprocess_audio(audio_path: str):
    audio, sample_rate = torchaudio.load(audio_path)

    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio, sample_rate, SAMPLE_RATE
        )

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)

    return audio.squeeze()


@torch.inference_mode()
def transcribe(audio_path: str) -> str:
    audio = preprocess_audio(audio_path)


    inputs = processor(
        audio.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        language="zh",
        task="transcribe",
    )


    return processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    args = parser.parse_args()

    print(transcribe(args.audio_path))
