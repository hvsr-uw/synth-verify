#!/usr/bin/env python3
"""Build the reward-only LoRA dataset (condition 3, fair variant).

Every prompting condition learns from the model's own experience plus a
reward signal. The original LoRA dataset (build_dataset.py) instead trains
on ORACLE trajectories, which quietly gives LoRA supervision the prompting
conditions never get. This builder removes that confound:

  - Roll out the plain no-memory agent on a TRAINING stream (seed 100,
    disjoint from the eval seeds 0/1/2) with sampling.
  - Keep only trajectories that earned reward 1.0 (rejection sampling,
    STaR-style), at most 2 successes per instance.
  - Balance value answers vs CANNOT_DETERMINE so the adapter does not just
    learn to abstain (or just to answer).
  - Emit the exact {"messages": [...]} format train_lora.py consumes,
    in 4 sequential chunks like the oracle dataset.

The oracle dataset stays as a labeled supervised ceiling, not the headline.

Usage (needs a served base model):
    python3 build_reward_dataset.py --n 300 --attempts 4 \
        --api-base http://localhost:8001/v1 --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from synth_verify.agent_harness import (  # noqa: E402
    LLMClient,
    MAX_TOOL_CALLS,
    MAX_TURNS,
    NoMemoryAgent,
    execute_calls,
    extract_answer,
)
from synth_verify.stream import score_instance  # noqa: E402
from synth_verify.task import CANNOT, generate_instances  # noqa: E402
from synth_verify.tools import ToolKit  # noqa: E402

N_CHUNKS = 4
MAX_KEEP_PER_INSTANCE = 2


def rollout(agent, view, toolkit, temperature):
    """One sampled attempt. Returns (messages, answer) with the full
    conversation, truncated at the answer-bearing assistant turn."""
    messages = agent._prompt_for(view)
    tool_calls_used = 0
    for _ in range(MAX_TURNS):
        out = agent.llm.chat(messages, temperature=temperature)
        messages.append({"role": "assistant", "content": out})
        results, n = execute_calls(out, toolkit)
        tool_calls_used += n
        answer = extract_answer(out)
        if answer is not None:
            return messages, answer
        if results:
            messages.append(
                {
                    "role": "user",
                    "content": "Tool results:\n%s\n\nContinue with more CALLs or give your ANSWER."
                    % results,
                }
            )
        if not results or tool_calls_used >= MAX_TOOL_CALLS:
            break
    # Last-chance turn, same as the harness.
    out = agent.llm.chat(
        messages
        + [{"role": "user", "content": "No more tool calls. Give your final ANSWER now."}],
        max_tokens=64,
        temperature=temperature,
    )
    messages.append({"role": "assistant", "content": out})
    return messages, extract_answer(out)


def harvest(agent, instances, toolkit, attempts, rng):
    """Rejection-sample successful trajectories. Returns kept examples
    as {"messages": [...], "type": t} dicts plus per-type stats."""
    kept = []
    stats = {}
    for inst in instances:
        successes = []
        for k in range(attempts):
            if len(successes) >= MAX_KEEP_PER_INSTANCE:
                break
            temp = 0.0 if k == 0 else 0.8
            try:
                messages, answer = rollout(
                    agent, inst.public_view(), toolkit, temp
                )
            except Exception:  # noqa: BLE001 - a failed attempt is just skipped
                continue
            if answer is None:
                continue
            reward = score_instance(inst, answer)
            st = stats.setdefault(
                inst.type, {"attempts": 0, "successes": 0, "kept": 0}
            )
            st["attempts"] += 1
            if reward == 1.0:
                st["successes"] += 1
                successes.append(messages)
        for messages in successes:
            kept.append({"messages": messages, "type": inst.type})
            stats[inst.type]["kept"] += 1
    return kept, stats


def balance(kept, rng, ratio=1.5):
    """Downsample the majority answer class to at most `ratio` x minority."""
    values = [e for e in kept if e["type"] == "A"]
    cannots = [e for e in kept if e["type"] != "A"]
    if not values or not cannots:
        return kept
    if len(values) > len(cannots) * ratio:
        rng.shuffle(values)
        values = values[: int(len(cannots) * ratio)]
    elif len(cannots) > len(values) * ratio:
        rng.shuffle(cannots)
        cannots = cannots[: int(len(values) * ratio)]
    out = values + cannots
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--api-base", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out-prefix", default="lora_reward")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    llm = LLMClient(args.api_base, args.model)
    probe = llm.chat(
        [{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=8,
    )
    print("endpoint probe reply: %r" % probe[:40], flush=True)

    agent = NoMemoryAgent(llm)
    toolkit = ToolKit(seed=args.seed)
    instances = generate_instances(args.n, seed=args.seed, toolkit=toolkit)
    order = list(range(len(instances)))
    rng.shuffle(order)
    instances = [instances[i] for i in order]

    kept, stats = harvest(agent, instances, toolkit, args.attempts, rng)
    print("harvest stats:", json.dumps(stats, indent=1), flush=True)
    kept = balance(kept, rng)
    n_cannot = sum(1 for e in kept if e["type"] != "A")
    print(
        "kept %d examples (%d value, %d CANNOT_DETERMINE)"
        % (len(kept), len(kept) - n_cannot, n_cannot),
        flush=True,
    )

    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    full_path = os.path.join(log_dir, "%s_dataset.jsonl" % args.out_prefix)
    with open(full_path, "w") as f:
        for ex in kept:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")
    per = max(1, len(kept) // N_CHUNKS)
    for i in range(N_CHUNKS):
        chunk = kept[i * per : (i + 1) * per] if i < N_CHUNKS - 1 else kept[i * per :]
        path = os.path.join(log_dir, "%s_chunk%d.jsonl" % (args.out_prefix, i))
        with open(path, "w") as f:
            for ex in chunk:
                f.write(json.dumps({"messages": ex["messages"]}) + "\n")
        print("chunk %d: %d examples -> %s" % (i, len(chunk), path), flush=True)
    print("DONE wrote", full_path)


if __name__ == "__main__":
    main()
