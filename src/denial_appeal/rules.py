"""Keyword-rule baseline: the 'base model' the trained classifier must beat."""
from __future__ import annotations

from .categories import CATEGORIES, CAUSE_INFO


def classify_rules(text: str) -> str:
    low = text.lower()
    best, best_score = "documentation_insufficient", -1
    for cause in CATEGORIES:
        score = sum(1 for kw in CAUSE_INFO[cause]["keywords"] if kw in low)
        # The synthetic denial code is a very strong rule signal.
        if CAUSE_INFO[cause]["denial_code"].lower() in low:
            score += 3
        if score > best_score:
            best, best_score = cause, score
    return best
