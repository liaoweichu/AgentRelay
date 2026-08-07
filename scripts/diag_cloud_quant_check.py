#!/usr/bin/env python3
"""Verify whether bnb_4bit quantization actually takes effect on the cloud
(12B Gemma4Unified) model. Loads the model with the SAME config used by the
gate, inspects the parameter dtypes / quantization layers, prints the memory
footprint, then exits WITHOUT running any episode (so no OOM risk from the
interaction loop)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.inference import HFModelExecutor, NativeGenerationConfig
from agentrelay.config import StorageLayout

CFG = {
    "model_id": "google/gemma-4-12b-it",
    "revision": "a69a4a15fe6a1b5a51373df662c1472be9b67683",
    "model_source": "modelscope",
    "dtype": "bfloat16",
    "quantization": "bnb_4bit",
    "device_map": "auto",
    "max_new_tokens": 1024,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "seed": 0,
    "architecture": "multimodal_lm",
    "enable_thinking": False,
    "local_files_only": True,
    "trust_remote_code": True,
}

import torch

storage = StorageLayout.from_env()
model = HFModelExecutor(NativeGenerationConfig(**CFG), storage).model

# count quantized params
n_params4bit = 0
n_full_bf16_linear = 0
n_all_params = 0
import bitsandbytes as bnb
from collections import Counter

dtype_counter = Counter()
for name, p in model.named_parameters():
    n_all_params += 1
    if isinstance(p, bnb.nn.Params4bit):
        n_params4bit += 1
    dtype_counter[str(p.dtype)] += 1

# count nn.Linear layers that are NOT quantized (suspicious if many remain bf16)
linear_bf16 = 0
linear_other = 0
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        if getattr(m, "weight", None) is not None and not isinstance(m.weight, bnb.nn.Params4bit):
            linear_bf16 += 1
        else:
            linear_other += 1

print("total params linalgs:", n_all_params)
print("Params4bit tensors:", n_params4bit)
print("dtype histogram:", dict(dtype_counter))
print("nn.Linear NOT quantized (bf16):", linear_bf16)
print("nn.Linear quantized/other:", linear_other)
try:
    print("model.get_memory_footprint():", model.get_memory_footprint() / 1e9, "GB")
except Exception as e:
    print("footprint err:", e)
print("GPU alloced:", torch.cuda.memory_allocated() / 1e9, "GB")
print("GPU reserved:", torch.cuda.memory_reserved() / 1e9, "GB")