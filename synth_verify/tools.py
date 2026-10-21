"""Deterministic synthetic tool environment.

Ten tools. Tools 1-6 share one underlying data store. Tools 7-10 are
independent cross-checks. Everything is deterministic given a single seed:
the store is built once with a seeded RNG, and per-key values that must not
depend on call order use a stable hash instead of RNG state.
"""

import hashlib
import random

ATTR_POOLS = {
    "color": ["red", "blue", "green", "amber"],
    "size": ["small", "medium", "large"],
    "origin": ["north", "south", "east", "west"],
    "code": ["A1", "B2", "C3", "D4"],
}
# Attributes no entity has; used for Type D (missing evidence) instances.
ABSENT_ATTRS = ["warranty", "serial", "batch", "supplier"]
CATEGORIES = ["alpha", "beta", "gamma"]
RELATIONS = ["connected_to", "derived_from", "sibling_of"]
N_ENTITIES = 30


def _stable_int(*parts):
    """Deterministic int from parts (hashlib, not salted builtin hash)."""
    h = hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


class ToolKit:
    """The 10 deterministic tools, bound to one seeded data store."""

    def __init__(self, seed=0):
        self.seed = seed
        rng = random.Random(seed)

        # ---- Primary store (shared by tools 1-6) ----
        self.entities = {}
        self.entity_category = {}
        names = ["entity_%02d" % i for i in range(N_ENTITIES)]
        for name in names:
            self.entity_category[name] = rng.choice(CATEGORIES)
            self.entities[name] = {
                attr: rng.choice(values) for attr, values in ATTR_POOLS.items()
            }

        # ---- Independent index (tools 7-10 side) ----
        # For each entity.attr key: the index either agrees with the primary
        # store, records a different value, or has no record at all.
        # These pools are what the instance generator samples from to build
        # Type A (agree), Type B (disagree) and Type C (index missing).
        self.index = {}
        self.index_status = {}
        for name, attrs in self.entities.items():
            for attr, val in attrs.items():
                key = "%s.%s" % (name, attr)
                roll = rng.random()
                if roll < 0.60:
                    self.index[key] = val
                    self.index_status[key] = "agree"
                elif roll < 0.80:
                    others = [v for v in ATTR_POOLS[attr] if v != val]
                    self.index[key] = rng.choice(others)
                    self.index_status[key] = "disagree"
                else:
                    self.index_status[key] = "missing"

        # ---- Relations (lookup_relation) ----
        self.relations = {}
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 : i + 4]:
                if n2 in self.entities and rng.random() < 0.5:
                    self.relations[(n1, n2)] = rng.choice(RELATIONS)

    # ---------- Tools 1-6: primary store ----------
    def get_entity(self, name):
        """Return all attributes of a named entity, or None."""
        attrs = self.entities.get(name)
        return dict(attrs) if attrs is not None else None

    def get_attribute(self, entity, attr):
        """Return one attribute value, or None if absent."""
        return self.entities.get(entity, {}).get(attr)

    def list_entities(self, category):
        """List entity names in a category."""
        return sorted(
            n for n, c in self.entity_category.items() if c == category
        )

    def compare_values(self, a, b):
        """Strict equality of two values."""
        return a == b

    def lookup_relation(self, e1, e2):
        """Return the recorded relation between two entities, or None."""
        return self.relations.get((e1, e2)) or self.relations.get((e2, e1))

    def count_entities(self, attr, value):
        """Count entities whose attribute equals value."""
        return sum(
            1 for attrs in self.entities.values() if attrs.get(attr) == value
        )

    # ---------- Tools 7-10: independent cross-checks ----------
    def get_source(self, entity, attr):
        """Provenance: which source recorded this value, or None."""
        if attr in self.entities.get(entity, {}):
            return "primary_registry"
        return None

    def check_consistency(self, v1, v2):
        """Equality with type normalization (case/whitespace)."""
        if v1 is None or v2 is None:
            return False
        return str(v1).strip().lower() == str(v2).strip().lower()

    def get_record_time(self, entity, attr):
        """Deterministic record timestamp, or None if absent."""
        if attr not in self.entities.get(entity, {}):
            return None
        days = _stable_int("time", entity, attr) % 365
        return "2026-%02d-%02dT08:00:00Z" % ((days % 12) + 1, (days % 28) + 1)

    def query_index(self, key):
        """Secondary index lookup, independent of tools 1-6. None if absent."""
        return self.index.get(key)
