"""Verb -> interaction-spec classification.

v1: keyword rules covering TRUMANS' templated vocabulary (S category only).
The LLM path (open-vocab) will emit the same spec schema and is drop-in.
"""
import re
from dataclasses import dataclass, asdict


@dataclass
class InteractionSpec:
    category: str            # "S" support-transfer | "R" reach | "T" traversal
    verb: str                # canonical verb: sit / lie / ...
    contact_part: str        # dominant contact body part
    commitment_posture: str  # posture at commitment
    facing_rule: str         # yaw semantics for construct()

    def to_dict(self):
        return asdict(self)


_SIT = InteractionSpec("S", "sit", "pelvis", "sit", "away_from_backrest")
_LIE = InteractionSpec("S", "lie", "torso", "lie", "along_major_axis")


def classify(text: str):
    """Return InteractionSpec for S-category transitions, else None (v1)."""
    t = text.lower().strip()
    # exclude states that merely mention sitting (e.g. "slide while sitting")
    if re.search(r"\bsit(s)? down\b", t) or t == "sit":
        return _SIT
    if re.search(r"\blie(s)? (down|face down)\b", t):
        return _LIE
    return None
