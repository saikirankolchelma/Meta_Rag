"""Single source of truth for the threshold-based tier routing rule.

Dependency-free leaf module: both dispatcher.py (the live gateway's cold-start
fallback) and decision_model.py (the EV model's band assignment and ultimate
fallback) import from here, so neither owns the other's constants.
"""

T1_DEFAULT = 0.4
T2_DEFAULT = 0.85


def score_to_tier(score: float, t1: float, t2: float) -> str:
    if score < t1:
        return "naive"
    if score < t2:
        return "parent"
    return "hyde"
