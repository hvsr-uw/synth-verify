#!/usr/bin/env python3
"""Build oracle-trajectory SFT dataset for the LoRA condition.

For each stream instance (seed 0), emits one multi-turn conversation:
  system (same BASE_SYSTEM the agent harness uses)
  user:    "Question: ... Use the tools if needed, then give your ANSWER."
  assistant: two canonical tool calls (get_attribute + query_index)
  user:    tool results (computed from the deterministic ToolKit)
  assistant: "ANSWER: <gold>"

These are oracle demonstrations of correct verification behavior:
the final answer follows the hidden rule (two independent confirmations
-> value, else CANNOT_DETERMINE). Training on them is behavior cloning
of the correct policy, NOT distillation of a stronger model.

Output: logs/lora_dataset.jsonl  (plus 4 sequential chunks logs/lora_chunk{i}.jsonl)
Run on Olvi (needs only CPU): python3 build_dataset.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from synth_verify.agent_harness import BASE_SYSTEM  # noqa: E402
from synth_verify.task import generate_instances  # noqa: E402
from synth_verify.tools import ToolKit  # noqa: E402

N = 200  # cond1 (100) + cond2 (100) streams worth of instances
SEED = 0
N_CHUNKS = 4


def oracle_messages(toolkit, inst):
    ev1 = toolkit.get_attribute(inst.entity, inst.attr)
    ev2 = toolkit.query_index("%s.%s" % (inst.entity, inst.attr))
    call_block = (
        "I'll look up the attribute in the primary records and cross-check "
        "it in the independent index.\n"
        "CALL: get_attribute(%s, %s)\nCALL: query_index(%s.%s)"
        % (inst.entity, inst.attr, inst.entity, inst.attr)
    )
    results_block = (
        "Tool results:\nCALL get_attribute(%s, %s) -> %r\n"
        "CALL query_index(%s.%s) -> %r\n\n"
        "Continue with more CALLs or give your ANSWER."
        % (inst.entity, inst.attr, ev1, inst.entity, inst.attr, ev2)
    )
    return [
        {"role": "system", "content": BASE_SYSTEM},
        {
            "role": "user",
            "content": "Question: %s\n\nUse the tools if needed, then give your ANSWER."
            % inst.question,
        },
        {"role": "assistant", "content": call_block},
        {"role": "user", "content": results_block},
        {"role": "assistant", "content": "ANSWER: %s" % inst.gold},
    ]


def main():
    toolkit = ToolKit(seed=SEED)
    instances = generate_instances(N, seed=SEED, toolkit=toolkit)
    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)

    examples = [{"messages": oracle_messages(toolkit, inst)} for inst in instances]
    # sanity: every gold is either a value or CANNOT_DETERMINE
    assert all(
        ex["messages"][-1]["content"].startswith("ANSWER: ") for ex in examples
    )
    n_cannot = sum(
        1 for ex in examples if ex["messages"][-1]["content"] == "ANSWER: CANNOT_DETERMINE"
    )
    with open(os.path.join(log_dir, "lora_dataset.jsonl"), "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    per = N // N_CHUNKS
    for i in range(N_CHUNKS):
        chunk = examples[i * per : (i + 1) * per]
        with open(os.path.join(log_dir, "lora_chunk%d.jsonl" % i), "w") as f:
            for ex in chunk:
                f.write(json.dumps(ex) + "\n")

    print(
        "wrote %d examples (%d CANNOT_DETERMINE) in %d chunks"
        % (len(examples), n_cannot, N_CHUNKS)
    )


if __name__ == "__main__":
    main()
