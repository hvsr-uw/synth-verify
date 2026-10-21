"""EBM-style belief memory for the synthetic verification task (condition 4).

Adapts Evolving Belief Memory (proposer / validator / persistent belief
state) to the hidden-rule stream:

- Belief state: propositions about what makes an answer trustworthy, each
  with a status (HYPOTHESIZED, SUPPORTED, CONTRADICTED) and evidence notes.
- Proposer: after each instance, proposes belief mutations from the latest
  trajectory + reward and the current beliefs. Sees reward only, NOT gold.
- Validator: judges each proposal against the cited evidence and returns
  SUPPORT / REFUTE / INCONCLUSIVE. Guards every belief update.
- Solver: the no-memory tool loop, with the current belief state in context.

Stdlib only. Wired into run_cond.py as --condition 4.
"""

import re

from .agent_harness import NoMemoryAgent

MAX_BELIEFS = 8

PROPOSER_SYSTEM = (
    "You study how to answer questions with tools. You keep short beliefs "
    "about what makes an answer trustworthy.\n"
    "Look at your current beliefs and your latest attempt, then propose updates.\n"
    "A belief earns SUPPORTED only if you have seen it work more than once.\n"
    "Mark a belief CONTRADICTED only if you saw it fail.\n"
    "Reply with 0 to 2 proposals, one per line starting with '- ', "
    "or the single word EMPTY if nothing new was learned.\n"
    "New belief format:\n"
    "- ADD: <one-sentence belief> | EVIDENCE: <concrete observations>\n"
    "Change format:\n"
    "- REVISE <number>: <HYPOTHESIZED|SUPPORTED|CONTRADICTED> | EVIDENCE: <concrete observations>"
)

VALIDATOR_SYSTEM = (
    "You judge proposed beliefs about answering questions with tools.\n"
    "For each numbered proposal give exactly one verdict: SUPPORT, REFUTE, "
    "or INCONCLUSIVE.\n"
    "SUPPORT needs more than one concrete observation in the evidence.\n"
    "REFUTE needs a concrete observation that contradicts the belief.\n"
    "Otherwise give INCONCLUSIVE.\n"
    "One line per proposal like: 1: SUPPORT - <short reason>\n"
    "If there are no proposals, reply with the single word NONE."
)

ADD_RE = re.compile(r"ADD:\s*(.*?)\s*\|\s*EVIDENCE:\s*(.*)", re.IGNORECASE)
REVISE_RE = re.compile(
    r"REVISE\s+(\d+)\s*:\s*(HYPOTHESIZED|SUPPORTED|CONTRADICTED)\s*\|\s*EVIDENCE:\s*(.*)",
    re.IGNORECASE,
)
VERDICT_RE = re.compile(r"\s*(\d+)\s*:\s*(SUPPORT|REFUTE|INCONCLUSIVE)\b", re.IGNORECASE)


def belief_text(beliefs):
    if not beliefs:
        return "Your beliefs about this task, learned from earlier questions:\n(empty - no beliefs yet)"
    lines = ["Your beliefs about this task, learned from earlier questions:"]
    for i, b in enumerate(beliefs, 1):
        lines.append(
            "%d. [%s] %s (evidence: %s)"
            % (i, b["status"], b["proposition"], b["evidence"])
        )
    return "\n".join(lines)


def parse_proposals(text):
    proposals = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:].strip()
        m = ADD_RE.match(body)
        if m:
            proposals.append(
                {
                    "op": "ADD",
                    "proposition": m.group(1).strip(),
                    "evidence": m.group(2).strip(),
                }
            )
            continue
        m = REVISE_RE.match(body)
        if m:
            proposals.append(
                {
                    "op": "REVISE",
                    "index": int(m.group(1)) - 1,
                    "status": m.group(2).upper(),
                    "evidence": m.group(3).strip(),
                }
            )
    return proposals[:2]


def parse_verdicts(text, n):
    verdicts = []
    for line in (text or "").splitlines():
        m = VERDICT_RE.match(line)
        if m and int(m.group(1)) <= n:
            verdicts.append(m.group(2).upper())
    while len(verdicts) < n:
        verdicts.append("INCONCLUSIVE")
    return verdicts[:n]


class EBMHarnessAgent(NoMemoryAgent):
    """Condition 4: EBM-style proposer/validator belief memory on 8B."""

    def __init__(self, llm):
        super().__init__(llm)
        self.name = "ebm"
        self.beliefs = []
        self.belief_history = []

    def _prompt_for(self, view, extra_system=""):
        extra = "\n\n" + belief_text(self.beliefs) + "\nFollow them when they apply."
        return super()._prompt_for(view, extra_system=extra)

    def reflect(self, view, trajectory, reward):
        """Proposer proposes, validator judges, belief state updates."""
        traj_text = "\n---\n".join(trajectory[-4:])
        proposer_messages = [
            {"role": "system", "content": PROPOSER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "%s\n\nLatest question: %s\n"
                    "Latest attempt (last steps):\n%s\n"
                    "Result: %s\n\nWhat belief updates do you propose?"
                    % (
                        belief_text(self.beliefs),
                        view["question"],
                        traj_text,
                        "CORRECT" if reward == 1.0 else "WRONG",
                    )
                ),
            },
        ]
        try:
            prop_out = self.llm.chat(proposer_messages, max_tokens=256)
        except Exception:  # noqa: BLE001 - a failed reflection skips the update
            self.belief_history.append(list(self.beliefs))
            return
        proposals = parse_proposals(prop_out)
        if not proposals:
            self.belief_history.append(list(self.beliefs))
            return
        numbered = "\n".join(
            "%d. %s: %s | EVIDENCE: %s"
            % (
                i + 1,
                p["op"],
                p.get("proposition", p.get("status")),
                p["evidence"],
            )
            for i, p in enumerate(proposals)
        )
        validator_messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM},
            {"role": "user", "content": "Proposals:\n%s" % numbered},
        ]
        try:
            val_out = self.llm.chat(validator_messages, max_tokens=128)
        except Exception:  # noqa: BLE001
            self.belief_history.append(list(self.beliefs))
            return
        for p, v in zip(proposals, parse_verdicts(val_out, len(proposals))):
            self._apply(p, v)
        self.belief_history.append(list(self.beliefs))

    def _apply(self, proposal, verdict):
        if proposal["op"] == "ADD":
            if verdict == "REFUTE":
                return  # validator guards: refuted beliefs never enter
            new_prop = proposal["proposition"].strip().lower()
            for b in self.beliefs:
                old_prop = b["proposition"].strip().lower()
                if new_prop == old_prop or new_prop in old_prop or old_prop in new_prop:
                    return  # already believed; skip duplicate
            status = "SUPPORTED" if verdict == "SUPPORT" else "HYPOTHESIZED"
            self.beliefs.append(
                {
                    "proposition": proposal["proposition"],
                    "status": status,
                    "evidence": proposal["evidence"],
                }
            )
        elif proposal["op"] == "REVISE":
            i = proposal["index"]
            if 0 <= i < len(self.beliefs):
                if verdict == "SUPPORT":
                    self.beliefs[i]["status"] = proposal["status"]
                elif verdict == "INCONCLUSIVE":
                    # inconclusive evidence weakens a supported belief
                    if self.beliefs[i]["status"] == "SUPPORTED":
                        self.beliefs[i]["status"] = "HYPOTHESIZED"
                self.beliefs[i]["evidence"] = proposal["evidence"]
        self._cap()

    def _cap(self):
        while len(self.beliefs) > MAX_BELIEFS:
            for j, b in enumerate(self.beliefs):
                if b["status"] == "CONTRADICTED":
                    del self.beliefs[j]
                    break
            else:
                for j, b in enumerate(self.beliefs):
                    if b["status"] == "HYPOTHESIZED":
                        del self.beliefs[j]
                        break
                else:
                    del self.beliefs[0]

    def final_belief_text(self):
        return belief_text(self.beliefs)
