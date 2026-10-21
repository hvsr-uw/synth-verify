"""Tool-calling agent harness for the synthetic verification task.

Runs against an OpenAI-compatible endpoint (vLLM). Agents:

- NoMemoryAgent: fresh context per instance. Loops: prompt model ->
  parse CALLs -> execute against ToolKit -> feed results back -> repeat
  until ANSWER or limits. No carried state.
- PlaybookAgent: same loop, but the prompt includes a playbook carried
  across instances. After each instance, Reflector proposes playbook
  updates from (trajectory, reward); Curator merges them into the
  playbook (capped at MAX_BULLETS).

Tool call format the model must emit:
    CALL: get_attribute(entity_03, color)
    CALL: query_index(entity_03.color)
Final answer format:
    ANSWER: red
    ANSWER: CANNOT_DETERMINE

Stdlib only (urllib) so it runs anywhere without extra deps.
"""

import json
import re
import urllib.request

CANNOT = "CANNOT_DETERMINE"
MAX_TOOL_CALLS = 6
MAX_TURNS = 4
MAX_BULLETS = 10

TOOL_DESCRIPTIONS = """You have these tools. Call them with: CALL: tool_name(arg1, arg2)
- get_entity(name): all attributes of an entity, e.g. CALL: get_entity(entity_03)
- get_attribute(entity, attr): one attribute value, e.g. CALL: get_attribute(entity_03, color)
- list_entities(category): entity names in a category
- compare_values(a, b): True if two values are equal
- lookup_relation(e1, e2): recorded relation between two entities, or nothing
- count_entities(attr, value): how many entities have an attribute value
- get_source(entity, attr): which source recorded a value, or nothing
- check_consistency(v1, v2): True if two values match (ignores case/space)
- get_record_time(entity, attr): when a value was recorded, or nothing
- query_index(key): independent second lookup, e.g. CALL: query_index(entity_03.color)
  Returns a value or nothing. This index was built separately from the main records.

When you are done, write your final answer as: ANSWER: <value>
or: ANSWER: CANNOT_DETERMINE
You may also write ANSWER: CANNOT_DETERMINE when the tools do not give you a clear answer."""

BASE_SYSTEM = (
    "You are a careful research assistant. Answer the question using the tools.\n"
    + TOOL_DESCRIPTIONS
)

TOOL_CALL_RE = re.compile(r"CALL\s*:\s*([a-z_]+)\s*\(([^)]*)\)")
ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)


class LLMClient:
    """Minimal OpenAI-compatible chat client over urllib."""

    def __init__(self, api_base, model, timeout=120):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, max_tokens=512, temperature=0.0, seed=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


def _parse_args(argstr):
    return [a.strip() for a in argstr.split(",") if a.strip()]


def execute_calls(text, toolkit):
    """Find CALLs in text, execute them, return (results_str, n_calls)."""
    out_lines = []
    n = 0
    for m in TOOL_CALL_RE.finditer(text):
        name, argstr = m.group(1), m.group(2)
        args = _parse_args(argstr)
        fn = getattr(toolkit, name, None)
        if fn is None or not callable(fn):
            out_lines.append("CALL %s -> ERROR: unknown tool" % name)
            continue
        try:
            res = fn(*args)
        except Exception as exc:  # noqa: BLE001 - tool misuse is model error
            out_lines.append("CALL %s(%s) -> ERROR: %s" % (name, argstr, exc))
            n += 1
            continue
        out_lines.append("CALL %s(%s) -> %r" % (name, argstr, res))
        n += 1
    return "\n".join(out_lines), n


def extract_answer(text):
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else None


class NoMemoryAgent:
    """Condition 1: fresh context per instance, no carried state."""

    def __init__(self, llm):
        self.llm = llm
        self.name = "no_memory"

    def _prompt_for(self, view, extra_system=""):
        user = "Question: %s\n\nUse the tools if needed, then give your ANSWER." % view["question"]
        return [
            {"role": "system", "content": BASE_SYSTEM + extra_system},
            {"role": "user", "content": user},
        ]

    def __call__(self, view, toolkit):
        messages = self._prompt_for(view)
        tool_calls_used = 0
        trajectory = []
        self.last_trajectory = trajectory
        for _ in range(MAX_TURNS):
            out = self.llm.chat(messages)
            trajectory.append(out)
            results, n = execute_calls(out, toolkit)
            tool_calls_used += n
            if results:
                messages.append({"role": "assistant", "content": out})
                messages.append(
                    {"role": "user", "content": "Tool results:\n%s\n\nContinue with more CALLs or give your ANSWER." % results}
                )
            answer = extract_answer(out)
            if answer is not None:
                return answer
            if not results or tool_calls_used >= MAX_TOOL_CALLS:
                break
        # Last chance: one more short turn asking only for the answer.
        try:
            out = self.llm.chat(
                messages
                + [{"role": "user", "content": "No more tool calls. Give your final ANSWER now."}],
                max_tokens=64,
            )
            trajectory.append(out)
            answer = extract_answer(out)
            if answer is not None:
                return answer
        except Exception:  # noqa: BLE001
            pass
        return ""


class PlaybookAgent(NoMemoryAgent):
    """Condition 2: ACE-style playbook carried across instances.

    Roles (all the same 8B model):
    - Generator: answers with playbook in context.
    - Reflector: after each instance, proposes bullet updates from
      (trajectory, reward). Sees reward only, NOT the gold label.
    - Curator: merges proposals into the playbook, capped at MAX_BULLETS.
    """

    def __init__(self, llm):
        super().__init__(llm)
        self.name = "playbook"
        self.playbook = []  # list of bullet strings
        self.playbook_history = []  # snapshot after each instance

    def _playbook_text(self):
        if not self.playbook:
            return "(playbook is empty)"
        return "\n".join("%d. %s" % (i + 1, b) for i, b in enumerate(self.playbook))

    def _prompt_for(self, view, extra_system=""):
        extra = (
            "\n\nYour playbook (lessons learned from earlier questions):\n%s\n"
            "Follow it when it applies." % self._playbook_text()
        )
        return super()._prompt_for(view, extra_system=extra)

    def reflect(self, view, trajectory, reward):
        """Reflector: propose playbook updates. Then Curator merges."""
        traj_text = "\n---\n".join(trajectory[-4:])
        reflector_prompt = [
            {
                "role": "system",
                "content": (
                    "You review a question-answering attempt and write short lessons "
                    "for a playbook. Rules for a good lesson: one sentence, concrete, "
                    "about WHEN to trust tool results. Bad: 'be careful'. "
                    "Good: 'If two tools give different values for the same thing, "
                    "answer CANNOT_DETERMINE.'\n"
                    "Current playbook:\n%s\n\n"
                    "Reply with 0-3 new or replacement lessons, one per line, "
                    "each starting with '- '. If nothing new was learned, reply EMPTY."
                    % self._playbook_text()
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question was: %s\nAttempt transcript (last steps):\n%s\n"
                    "Result: %s\n\nWhat lessons should go in the playbook?"
                    % (
                        view["question"],
                        traj_text,
                        "CORRECT" if reward == 1.0 else "WRONG",
                    )
                ),
            },
        ]
        try:
            out = self.llm.chat(reflector_prompt, max_tokens=256)
        except Exception:  # noqa: BLE001
            return
        proposals = [
            line[2:].strip()
            for line in out.splitlines()
            if line.strip().startswith("- ") and len(line.strip()) > 4
        ]
        if not proposals or out.strip().upper() == "EMPTY":
            self.playbook_history.append(list(self.playbook))
            return
        # Curator: append, dedup (case-insensitive), cap at MAX_BULLETS (newest).
        seen = {b.strip().lower() for b in self.playbook}
        for p in proposals:
            if p.strip().lower() not in seen:
                self.playbook.append(p)
                seen.add(p.strip().lower())
        self.playbook = self.playbook[-MAX_BULLETS:]
        self.playbook_history.append(list(self.playbook))

    def final_playbook_text(self):
        return self._playbook_text()
