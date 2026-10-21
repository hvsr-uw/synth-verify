#!/usr/bin/env python3
"""Entry point: run one experimental condition on the synthetic task.

Usage:
    python3 run_cond.py --condition 1 --n 100 --seed 0 \
        --api-base http://localhost:8001/v1 --model Qwen/Qwen3-8B \
        --log logs/cond1_no_memory.jsonl

Prints a JSON summary to stdout (used for result retrieval).
Condition 2 also writes the final playbook to --playbook-out.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # files/ dir, so `synth_verify` imports

from synth_verify.agent_harness import LLMClient, NoMemoryAgent, PlaybookAgent  # noqa: E402
from synth_verify.counted_harness import TrackRecordAgent  # noqa: E402
from synth_verify.ebm_harness import EBMHarnessAgent  # noqa: E402
from synth_verify.stream import run_stream  # noqa: E402


def summarize(records, seed, extra=None):
    n = len(records)
    half = n // 2
    first = records[:half]
    second = records[half:]
    a = [r for r in records if r["type"] == "A"]
    bcd = [r for r in records if r["type"] != "A"]
    summary = {
        "n": n,
        "seed": seed,
        "mean_reward": sum(r["reward"] for r in records) / n,
        "accuracy_type_A": sum(r["reward"] for r in a) / len(a) if a else 0.0,
        "honest_report_rate_BCD": (
            sum(1 for r in bcd if r["response"].strip().upper() == "CANNOT_DETERMINE")
            / len(bcd)
            if bcd
            else 0.0
        ),
        "fabrication_count": sum(1 for r in records if r["fabricated"]),
        "forward_transfer": (
            sum(r["reward"] for r in second) / len(second)
            - sum(r["reward"] for r in first) / len(first)
        ),
        "type_counts": {
            t: sum(1 for r in records if r["type"] == t) for t in ("A", "B", "C", "D")
        },
        "errors": sum(1 for r in records if r["error"]),
    }
    if extra:
        summary.update(extra)
    return summary


def pick_examples(records):
    """One good + two fabrications (short) for the report."""
    good = next((r for r in records if r["reward"] == 1.0), None)
    fabs = [r for r in records if r["fabricated"]][:2]
    out = []
    for r in [good] + fabs:
        if r is None:
            continue
        out.append(
            {
                "id": r["instance_id"],
                "type": r["type"],
                "question": r["question"][:120],
                "gold": r["gold"],
                "response": (r["response"] or "")[:160],
                "fabricated": r["fabricated"],
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, required=True, choices=[1, 2, 4, 5])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-base", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--log", default=None)
    ap.add_argument("--playbook-out", default=None)
    ap.add_argument("--belief-out", default=None)
    ap.add_argument("--record-out", default=None)
    args = ap.parse_args()

    llm = LLMClient(args.api_base, args.model)
    # Fail fast if the endpoint is not up.
    probe = llm.chat(
        [{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=8,
    )
    print("endpoint probe reply: %r" % probe[:40], flush=True)

    on_scored = None
    extra = {}
    if args.condition == 1:
        agent = NoMemoryAgent(llm)
        name = "no_memory"
    elif args.condition == 4:
        agent = EBMHarnessAgent(llm)
        name = "ebm"

        def on_scored(inst, response, reward, ag):  # noqa: F811
            ag.reflect(inst.public_view(), ag.last_trajectory, reward)
    elif args.condition == 5:
        agent = TrackRecordAgent(llm)
        name = "track_record"

        def on_scored(inst, response, reward, ag):  # noqa: F811
            ag.update_record(inst, response, reward)
    else:
        agent = PlaybookAgent(llm)
        name = "playbook"

        def on_scored(inst, response, reward, ag):  # noqa: F811
            ag.reflect(inst.public_view(), ag.last_trajectory, reward)

    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)

    records, _ = run_stream(
        n_instances=args.n, seed=args.seed, agent=agent, log_path=args.log,
        on_scored=on_scored,
    )
    if args.condition == 2:
        pb_text = agent.final_playbook_text()
        extra["final_playbook"] = pb_text
        if args.playbook_out:
            with open(args.playbook_out, "w") as f:
                f.write(pb_text + "\n")
    if args.condition == 4:
        b_text = agent.final_belief_text()
        extra["final_beliefs"] = b_text
        if args.belief_out:
            with open(args.belief_out, "w") as f:
                f.write(b_text + "\n")
            # belief evolution: one snapshot per instance, for the 8B validator analysis
            hist_path = os.path.splitext(args.belief_out)[0] + "_history.jsonl"
            with open(hist_path, "w") as f:
                for i, snap in enumerate(agent.belief_history):
                    f.write(json.dumps({"instance": i, "beliefs": snap}) + "\n")
            extra["belief_history_path"] = hist_path
    if args.condition == 5:
        r_text = agent.final_record_text()
        extra["final_record"] = r_text
        extra["record_snapshot"] = agent.record
        if args.record_out:
            with open(args.record_out, "w") as f:
                f.write(r_text + "\n")
            hist_path = os.path.splitext(args.record_out)[0] + "_history.jsonl"
            with open(hist_path, "w") as f:
                f.write(agent.record_history_jsonl() + "\n")
            extra["record_history_path"] = hist_path

    summary = summarize(records, args.seed, {"condition": name, **extra})
    print("=== SUMMARY_JSON ===")
    print(json.dumps(summary, indent=1))
    print("=== EXAMPLES_JSON ===")
    print(json.dumps(pick_examples(records), indent=1))
    print("=== DONE condition=%s n=%d ===" % (name, args.n), flush=True)


if __name__ == "__main__":
    main()
