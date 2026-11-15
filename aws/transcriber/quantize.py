"""Build-time step: make an 8-bit copy of the model so Lambda never holds the 3 GB float32 version.

Dynamic int8 quantization stores the weights of every Linear layer as 8-bit integers (activations stay
float). On CPU this is roughly 4x smaller and much faster. Converting needs the float32 model and the
int8 copy in memory together (~4 GB), which is more than a 3 GB Lambda has, so it's done here, once.
"""
import os
import sys

import torch
from transformers import AutoModelForSpeechSeq2Seq

src, dst = sys.argv[1], sys.argv[2]
# oneDNN, not QNNPACK: QNNPACK fails to start inside real Lambda ("failed to initialize QNNPACK"), although it works
# in the local emulator. The engine used here must match the one app.py sets before loading.
torch.backends.quantized.engine = os.environ.get("QUANT_ENGINE", "onednn")
model = AutoModelForSpeechSeq2Seq.from_pretrained(src).eval()
model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(model, dst)
print(f"saved int8 model to {dst} (engine: {torch.backends.quantized.engine})")
