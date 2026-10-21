"""Instance generation for the synthetic verification task.

Hidden rule (constant across the stream, never told to the model):
an answer is VALID only if at least two independent tools confirm the same
value. Otherwise the correct response is CANNOT_DETERMINE.

Instance types (hidden from the model):
  A (~40%): two independent tools confirm the same value -> gold = the value
  B (~20%): two tools disagree                        -> gold = CANNOT_DETERMINE
  C (~20%): only one tool has evidence                 -> gold = CANNOT_DETERMINE
  D (~20%): no tool has evidence                       -> gold = CANNOT_DETERMINE
"""

import random
from dataclasses import dataclass

from .tools import ABSENT_ATTRS

CANNOT = "CANNOT_DETERMINE"

# 2-3 paraphrase templates per type. Kept stylistically similar across types
# so the instance type cannot be read off the phrasing alone.
TEMPLATES = {
    "A": [
        "What is the {attr} of {entity}?",
        "Report the {attr} for {entity}.",
        "Find the {attr} of {entity}.",
    ],
    "B": [
        "What is the {attr} of {entity}?",
        "Give the {attr} of {entity}.",
        "What value is recorded for the {attr} of {entity}?",
    ],
    "C": [
        "What is the {attr} of {entity}?",
        "State the {attr} of {entity}.",
        "Look up the {attr} of {entity}.",
    ],
    "D": [
        "What is the {attr} of {entity}?",
        "Report the {attr} of {entity}.",
        "Determine the {attr} of {entity}.",
    ],
}


@dataclass
class Instance:
    id: int
    type: str  # "A" | "B" | "C" | "D"
    entity: str
    attr: str
    question: str
    gold: str  # value string, or CANNOT_DETERMINE

    def public_view(self):
        """What the agent is allowed to see. Gold and type stay hidden."""
        return {
            "id": self.id,
            "question": self.question,
            "entity": self.entity,
            "attr": self.attr,
        }


def _pool(toolkit, status):
    return [
        key for key, s in toolkit.index_status.items() if s == status
    ]


def generate_instances(n, seed, toolkit):
    """Build n instances with exact 40/20/20/20 type proportions."""
    rng = random.Random(seed + 1)  # separate stream from toolkit construction

    n_a = int(n * 0.4)
    n_b = int(n * 0.2)
    n_c = int(n * 0.2)
    n_d = n - n_a - n_b - n_c
    counts = {"A": n_a, "B": n_b, "C": n_c, "D": n_d}

    pool_a = _pool(toolkit, "agree")
    pool_b = _pool(toolkit, "disagree")
    pool_c = _pool(toolkit, "missing")
    assert pool_a and pool_b and pool_c, "index pools must be non-empty"

    entity_names = sorted(toolkit.entities.keys())
    instances = []
    inst_id = 0
    for itype, count in counts.items():
        for _ in range(count):
            if itype == "D":
                entity = rng.choice(entity_names)
                attr = rng.choice(ABSENT_ATTRS)
                gold = CANNOT
            else:
                pool = {"A": pool_a, "B": pool_b, "C": pool_c}[itype]
                key = rng.choice(pool)
                entity, attr = key.split(".")
                if itype == "A":
                    gold = str(toolkit.get_attribute(entity, attr))
                else:
                    gold = CANNOT
            template = rng.choice(TEMPLATES[itype])
            question = template.format(entity=entity, attr=attr)
            instances.append(
                Instance(
                    id=inst_id,
                    type=itype,
                    entity=entity,
                    attr=attr,
                    question=question,
                    gold=gold,
                )
            )
            inst_id += 1
    return instances
