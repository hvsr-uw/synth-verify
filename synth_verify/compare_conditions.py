#!/usr/bin/env python3
"""Compare all conditions from their SUMMARY_JSON blocks.

Reads the .out files produced by the runs, extracts each summary, and
prints a markdown table plus the key contrasts for the report:
prompting (ACE playbook) vs EBM-style beliefs vs LoRA, all against the
no-memory baseline, on the same 8B model.

Usage:
    python3 compare_conditions.py logs/cond1_no_memory.out \
        logs/cond2_playbook.out logs/cond4_ebm.out [more .out files ...]
"""

import json
import re
import sys

SUMMARY_RE = re.compile(r"=== SUMMARY_JSON ===\s*(\{.*?\})\s*=== EXAMPLES_JSON ===", re.DOTALL)

ROWS = [
    ("mean_reward", "Mean reward", "%.3f"),
    ("accuracy_type_A", "Accuracy (answerable)", "%.3f"),
    ("honest_report_rate_BCD", "Honest report rate (B/C/D)", "%.3f"),
    ("fabrication_count", "Fabrications", "%d"),
    ("forward_transfer", "Forward transfer", "%+.3f"),
    ("errors", "Errors", "%d"),
]


def load_summary(path):
    with open(path) as f:
        text = f.read()
    m = SUMMARY_RE.search(text)
    if not m:
        return None
    return json.loads(m.group(1))


def main():
    summaries = {}
    for path in sys.argv[1:]:
        s = load_summary(path)
        if s is None:
            print("WARN: no SUMMARY_JSON in %s" % path, file=sys.stderr)
            continue
        summaries[s.get("condition", path)] = s
    if not summaries:
        print("No summaries found.")
        return
    names = list(summaries)
    print("| Metric | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for key, label, fmt in ROWS:
        cells = []
        for name in names:
            v = summaries[name].get(key)
            cells.append(fmt % v if isinstance(v, (int, float)) else str(v))
        print("| %s | %s |" % (label, " | ".join(cells)))
    base = summaries.get("no_memory")
    if base:
        print()
        for name in names:
            if name == "no_memory":
                continue
            s = summaries[name]
            print(
                "%s vs no-memory: reward %+.3f, honest rate %+.3f, fabrications %+d"
                % (
                    name,
                    s["mean_reward"] - base["mean_reward"],
                    s["honest_report_rate_BCD"] - base["honest_report_rate_BCD"],
                    s["fabrication_count"] - base["fabrication_count"],
                )
            )


if __name__ == "__main__":
    main()
