#!/usr/bin/env python3
"""Sequential LoRA SFT on oracle trajectories (condition 3).

Trains Qwen3-8B with PEFT LoRA in N_CHUNKS sequential stages, each stage
continuing from the previous adapter (streaming-style updates, batched per
chunk for weekend feasibility). Response-only loss: only the final
assistant turn (the ANSWER) contributes to the loss.

Saves the final adapter to --out (default logs/lora_adapter).
Prints per-chunk mean loss. CPU/GPU: needs 1x A100-40GB.

Usage:
    source /data/hvardhan2/lora-env/bin/activate
    python3 train_lora.py --chunks logs/lora_chunk0.jsonl logs/lora_chunk1.jsonl ...
"""

import argparse
import json
import os
import random
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
CACHE = "/data/hvardhan2/hf-cache"


def set_seeds(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_examples(path):
    with open(path) as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def encode_example(tok, messages, max_len=2048):
    """Tokenize; labels = -100 everywhere except the final assistant turn."""
    prompt_text = tok.apply_chat_template(messages[:-1], tokenize=False)
    full_text = tok.apply_chat_template(messages, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, truncation=True, max_length=max_len,
                   add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[: len(full_ids)]
    assert len(labels) == len(full_ids)
    return {"input_ids": full_ids, "labels": labels}


def train_chunk(model, tok, examples, epochs=3, lr=2e-4):
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    losses = []
    for _ in range(epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), 2):  # micro-batch 2
            batch = examples[i : i + 2]
            enc = [encode_example(tok, m) for m in batch]
            maxl = max(len(e["input_ids"]) for e in enc)
            input_ids, labels, mask = [], [], []
            for e in enc:
                pad = maxl - len(e["input_ids"])
                input_ids.append(e["input_ids"] + [tok.pad_token_id] * pad)
                labels.append(e["labels"] + [-100] * pad)
                mask.append([1] * len(e["input_ids"]) + [0] * pad)
            input_ids = torch.tensor(input_ids, device=model.device)
            labels = torch.tensor(labels, device=model.device)
            mask = torch.tensor(mask, device=model.device)
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seeds(args.seed)

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "lora_adapter"
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    for ci, cpath in enumerate(args.chunks):
        examples = load_examples(cpath)
        mean_loss = train_chunk(model, tok, examples, epochs=args.epochs)
        print("chunk %d/%d: n=%d mean_loss=%.4f" % (ci + 1, len(args.chunks), len(examples), mean_loss), flush=True)
        ckpt = out_dir + "_chunk%d" % ci
        model.save_pretrained(ckpt)
        print("saved", ckpt, flush=True)

    model.save_pretrained(out_dir)
    print("FINAL ADAPTER:", out_dir)


if __name__ == "__main__":
    main()
