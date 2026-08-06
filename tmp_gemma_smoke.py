#!/usr/bin/env python3
"""Smoke test loading a Gemma 4 model from its ModelScope snapshot dir."""
import argparse
import sys
from pathlib import Path

import torch

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    snap = Path(args.snapshot)
    if not (snap / "config.json").exists():
        print(f"NO_CONFIG: {snap}")
        return 2
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    print("loading processor...")
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    print(f"processor ok: {type(proc).__name__}")

    print("loading model...")
    kwargs = {"local_files_only": True}
    if args.cuda:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": torch.device("cpu")}
    try:
        model = AutoModelForMultimodalLM.from_pretrained(snap, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"LOAD_FAILED: {type(exc).__name__}: {exc}")
        return 3
    model.eval()
    print(f"model ok: {type(model).__name__}")

    messages = [
        {"role": "system", "content": "You are a helpful agent. Act via executable actions only."},
        {"role": "user", "content": "Observation: You are in a room. Valid actions: go to 1, take 1. Next action:"},
    ]
    prompt = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = proc(text=prompt, return_tensors="pt")
    if args.cuda:
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=32,
            do_sample=False,
        )
    text = proc.decode(out[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"GENERATED: {text!r}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())