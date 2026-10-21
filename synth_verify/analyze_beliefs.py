#!/usr/bin/env python3
"""Analyze the EBM (condition 4) run: does the validator hold up at 8B?

Two views:
1. Validator-assumption stress test (8B analog of the EBM paper's belief
   correctness analysis): the paper's Figure 5b shows SUPPORTED beliefs are
   predominantly correct at 27B+. Here we test the same idea behaviorally:
   mean reward on instances answered while a SUPPORTED belief was present
   vs only HYPOTHESIZED vs no beliefs. If SUPPORTED does not beat
   HYPOTHESIZED at 8B, the validator is not tracking truth at this scale -
   which is exactly what fabrication would predict.

2. Belief evolution timeline (HTML): every belief's lifecycle across the
   stream - when it was proposed, and how its status changed.

Usage:
    python3 analyze_beliefs.py --trace logs/cond4_trace.jsonl \
        --history logs/cond4_final_beliefs_history.jsonl \
        --out logs/cond4_belief_analysis.html
"""

import argparse
import html
import json
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def status_at_answer(history, i):
    """Belief snapshot in effect when instance i was answered.

    reflect() appends a snapshot AFTER each instance, so the state used to
    answer instance i is history[i-1] (empty for i == 0).
    """
    if i == 0 or not history:
        return []
    return history[i - 1]["beliefs"] if i - 1 < len(history) else history[-1]["beliefs"]


def stress_test(trace, history):
    groups = {"supported_present": [], "hypothesized_only": [], "no_beliefs": []}
    for i, rec in enumerate(trace):
        beliefs = status_at_answer(history, i)
        statuses = {b["status"] for b in beliefs}
        if "SUPPORTED" in statuses:
            groups["supported_present"].append(rec["reward"])
        elif beliefs:
            groups["hypothesized_only"].append(rec["reward"])
        else:
            groups["no_beliefs"].append(rec["reward"])
    out = {}
    for k, rewards in groups.items():
        out[k] = {
            "n": len(rewards),
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        }
    sup = out["supported_present"]["mean_reward"]
    hyp = out["hypothesized_only"]["mean_reward"]
    out["validator_gap"] = (
        sup - hyp if (sup is not None and hyp is not None) else None
    )
    return out


def lifecycles(history):
    """Track each distinct proposition across snapshots."""
    tracks = {}  # proposition -> list of (instance, status)
    for snap in history:
        for b in snap["beliefs"]:
            key = b["proposition"].strip().lower()
            tracks.setdefault(key, []).append((snap["instance"], b["status"]))
    return tracks


def transitions(history):
    counts = defaultdict(int)
    prev = {}
    for snap in history:
        cur = {b["proposition"].strip().lower(): b["status"] for b in snap["beliefs"]}
        for prop, st in cur.items():
            if prop not in prev:
                counts["proposed"] += 1
                counts["entered_" + st] += 1
            elif prev[prop] != st:
                counts[prev[prop] + "->" + st] += 1
        prev = cur
    return dict(counts)


def render_html(trace, history, test):
    tracks = lifecycles(history)
    trans = transitions(history)
    n = len(trace)
    color = {"SUPPORTED": "#2e7d32", "HYPOTHESIZED": "#f9a825", "CONTRADICTED": "#c62828"}
    rows = []
    for prop, events in tracks.items():
        by_inst = {i: s for i, s in events}
        cells = []
        last = None
        for i in range(n):
            st = by_inst.get(i, last)
            last = st
            bg = color.get(st, "#eeeeee") if st else "#fafafa"
            cells.append('<td style="background:%s" title="%s"></td>' % (bg, st or ""))
        rows.append(
            "<tr><td class='prop'>%s</td>%s</tr>" % (html.escape(prop[:90]), "".join(cells))
        )
    gap = test["validator_gap"]
    gap_txt = "n/a" if gap is None else "%+.3f" % gap
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>EBM at 8B: belief analysis</title>
<style>body{font-family:sans-serif;margin:24px} table{border-collapse:collapse}
td{width:8px;height:14px;border:1px solid #fff} td.prop{width:auto;font-size:12px;padding-right:8px}
.kpi{display:inline-block;margin-right:24px} .big{font-size:22px;font-weight:bold}</style>
</head><body>
<h2>EBM-style belief memory at 8B: validator stress test</h2>
<p>The EBM paper (27B+) shows SUPPORTED beliefs are predominantly correct.
At 8B, does a SUPPORTED belief actually predict better answers?</p>
<div class="kpi">Validator gap (supported minus hypothesized reward)<br><span class="big">%s</span></div>
<div class="kpi">Supported present<br><span class="big">%s</span> (n=%d)</div>
<div class="kpi">Hypothesized only<br><span class="big">%s</span> (n=%d)</div>
<div class="kpi">No beliefs<br><span class="big">%s</span> (n=%d)</div>
<h3>Belief lifecycles (green=supported, yellow=hypothesized, red=contradicted)</h3>
<table>%s</table>
<h3>Status transitions</h3><pre>%s</pre>
</body></html>""" % (
        gap_txt,
        _fmt(test["supported_present"]["mean_reward"]), test["supported_present"]["n"],
        _fmt(test["hypothesized_only"]["mean_reward"]), test["hypothesized_only"]["n"],
        _fmt(test["no_beliefs"]["mean_reward"]), test["no_beliefs"]["n"],
        "".join(rows),
        html.escape(json.dumps(trans, indent=1)),
    )


def _fmt(x):
    return "n/a" if x is None else "%.3f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    trace = load_jsonl(args.trace)
    history = load_jsonl(args.history)
    test = stress_test(trace, history)
    print(json.dumps(test, indent=1))
    print("transitions:", json.dumps(transitions(history)))
    if args.out:
        with open(args.out, "w") as f:
            f.write(render_html(trace, history, test))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
