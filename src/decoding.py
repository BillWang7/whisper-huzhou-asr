"""Decoding settings shared by the demo and the evaluation scripts.

Both the fine-tuned model and baseline Whisper are decoded with exactly these settings.

Loop guard: Whisper sometimes gets stuck repeating one character or phrase until it hits the length
limit. Following OpenAI's Whisper, if an output compresses suspiciously well (a sign of repetition),
it is thrown away and decoded again with a little randomness (temperature 0.2, then 0.4, ...). If every
attempt still loops, the last attempt is kept. 1.35 is Hugging Face's recommended threshold for its
token-based compression ratio; on our test set it never fires on a human transcript.
"""
import torch

GENERATE_KWARGS = dict(
    language="zh",
    task="transcribe",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=1.35,
)

SEED = 0  # makes the random retries repeatable


def pick_device() -> torch.device:
    """NVIDIA GPU if available, then Apple GPU, otherwise CPU (always CPU inside Docker on a Mac)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate(model, input_features, attention_mask):
    torch.manual_seed(SEED)
    return model.generate(input_features, attention_mask=attention_mask, **GENERATE_KWARGS)
