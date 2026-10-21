#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the base model for vLLM serving (cond3).

The LoRA adapter IS the continual-learning memory for condition 3, so the
eval harness is the plain no-memory agent: any improvement over cond1 comes
from the weights, not the prompt.

Usage (on Olvi, needs lora-env + GPU):
    source /data/hvardhan2/lora-env/bin/activate
    python3 merge_lora.py --adapter logs/lora_adapter --out logs/merged_8b_lora
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
CACHE = "/data/hvardhan2/hf-cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="trained LoRA adapter dir")
    ap.add_argument("--out", required=True, help="output dir for merged model")
    args = ap.parse_args()

    print("loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("loading adapter...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging adapter into base...", flush=True)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    print("saving merged model...", flush=True)
    model.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE, trust_remote_code=True
    )
    tok.save_pretrained(args.out)
    print("MERGED to", args.out, flush=True)


if __name__ == "__main__":
    main()
