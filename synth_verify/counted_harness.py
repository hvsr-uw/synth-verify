"""Condition 5: track-record memory — memory counted, not written.

The method in one paragraph: no model writes or judges this memory. After
each instance, deterministic code classifies the situation the model faced,
using only the observable state of the two independent sources (the primary
store via get_attribute, the independent index via query_index): both held
the same value (agree), both held different values (conflict), exactly one
held a value (single), neither held one (none). Code then counts the model's
own record per situation — how often it faced it, how often it answered
with a value, how often it was correct — and the next prompt shows those
exact counts. The memory cannot be fabricated, because arithmetic wrote it.

Classification never reads the instance's gold or type. It reads the two
source values, which is what the situation IS. (A unit test checks the
classification against the hidden types on generated instances: agree=A,
conflict=B, single=C, none=D.)
"""

import copy
import json

from .agent_harness import NoMemoryAgent
from .task import CANNOT

SITUATIONS = ("agree", "conflict", "single", "none")
SITUATION_LABELS = {
    "agree": "two tools gave the same value",
    "conflict": "two tools gave different values",
    "single": "only one tool had a value",
    "none": "no tool had a value",
}


def classify_situation(tools, entity, attr):
    """Sort an instance into a situation from the two sources' state."""
    record_val = tools.get_attribute(entity, attr)
    index_val = tools.query_index("%s.%s" % (entity, attr))
    if record_val is not None and index_val is not None:
        return "agree" if str(record_val) == str(index_val) else "conflict"
    if record_val is not None or index_val is not None:
        return "single"
    return "none"


class TrackRecordAgent(NoMemoryAgent):
    """No-memory agent plus a code-counted record of its own outcomes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools = None
        self.record = {
            s: {"faced": 0, "answered": 0, "correct": 0} for s in SITUATIONS
        }
        self.record_history = []

    def __call__(self, view, tools):
        self._tools = tools
        return super().__call__(view, tools)

    def _prompt_for(self, view, extra_system=""):
        return super()._prompt_for(view, extra_system=self.record_block())

    def record_block(self):
        total = sum(c["faced"] for c in self.record.values())
        if total == 0:
            return ""
        lines = [
            "\n\nYour track record so far on this task, counted exactly "
            "from your own earlier answers:\n"
        ]
        for s in SITUATIONS:
            c = self.record[s]
            if c["faced"] == 0:
                continue
            lines.append(
                "- When %s: faced %d, answered with a value %d, correct %d.\n"
                % (SITUATION_LABELS[s], c["faced"], c["answered"], c["correct"])
            )
        lines.append(
            "These counts are updated after every question. "
            "They are exact, not advice.\n"
        )
        return "".join(lines)

    def update_record(self, inst, response, reward):
        """Called after scoring. Counts what happened. Returns the situation."""
        if self._tools is None:
            return None
        sit = classify_situation(self._tools, inst.entity, inst.attr)
        r = (response or "").strip()
        answered = r != "" and r.upper() != CANNOT
        cell = self.record[sit]
        cell["faced"] += 1
        if answered:
            cell["answered"] += 1
        if reward == 1.0:
            cell["correct"] += 1
        self.record_history.append(
            {
                "after_instance": len(self.record_history),
                "situation": sit,
                "answered": answered,
                "reward": reward,
                "record": copy.deepcopy(self.record),
            }
        )
        return sit

    def final_record_text(self):
        block = self.record_block()
        return block if block else "(no instances faced)"

    def record_history_jsonl(self):
        return "\n".join(json.dumps(h) for h in self.record_history)
