#!/usr/bin/env python3
"""Verify that bnb_4bit quantization takes effect on the cloud model."""

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import StorageLayout
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig

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

def main() -> int:
    """Load the fixed cloud profile and print quantization diagnostics."""

    import bitsandbytes as bnb
    import torch

    storage = StorageLayout.from_env()
    model = HFModelExecutor(NativeGenerationConfig(**CFG), storage).model

    n_params4bit = 0
    n_all_params = 0
    dtype_counter = Counter()
    for parameter in model.parameters():
        n_all_params += 1
        if isinstance(parameter, bnb.nn.Params4bit):
            n_params4bit += 1
        dtype_counter[str(parameter.dtype)] += 1

    linear_bf16 = 0
    linear_other = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weight = getattr(module, "weight", None)
            if weight is not None and not isinstance(weight, bnb.nn.Params4bit):
                linear_bf16 += 1
            else:
                linear_other += 1

    print("total parameter tensors:", n_all_params)
    print("Params4bit tensors:", n_params4bit)
    print("dtype histogram:", dict(dtype_counter))
    print("nn.Linear NOT quantized (bf16):", linear_bf16)
    print("nn.Linear quantized/other:", linear_other)
    if hasattr(model, "get_memory_footprint"):
        print("model.get_memory_footprint():", model.get_memory_footprint() / 1e9, "GB")
    print("GPU allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
    print("GPU reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
