"""Which fine-tuned model the demo and the evaluation scripts load: one place to change.

Pinned to an exact Hugging Face commit, so uploading a new model to the repo can't silently change the demo or the
published numbers. v1 = the original 2024 model (also tagged "v1" on Hugging Face and on GitHub).
The AWS site has its own pin: MODEL_REVISION in aws/transcriber/Dockerfile.ct2 (keep the two the same).
"""
MODEL_ID = "BillWang7/whisper-medium-ch-hz"
MODEL_REVISION = "610b9056415e3b4251792d65225c27050125ae41"  # v1
