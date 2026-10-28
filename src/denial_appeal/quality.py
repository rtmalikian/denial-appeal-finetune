"""Appeal-draft quality checks.

The hallucination checker is deliberately simple and auditable: any denial
code (DEN-###) or policy number (Policy X.Y.Z) that appears in an appeal but
NOT in the source denial letter is flagged as invented.
"""
from __future__ import annotations

import re

from .categories import CAUSE_INFO

_CODE_RE = re.compile(r"\bDEN-\d+\b")
_POLICY_RE = re.compile(r"\bPolicy\s+\d+(?:\.\d+)+\b", re.IGNORECASE)


def extract_codes(text: str) -> set[str]:
    return set(_CODE_RE.findall(text))


def extract_policy_numbers(text: str) -> set[str]:
    return {m.group(0) for m in _POLICY_RE.finditer(text)}


def check_appeal(denial_letter: str, appeal_letter: str, expected_cause: str | None = None) -> dict:
    invented_codes = extract_codes(appeal_letter) - extract_codes(denial_letter)
    invented_policies = extract_policy_numbers(appeal_letter) - extract_policy_numbers(denial_letter)
    cited = False
    if expected_cause is not None:
        cited = CAUSE_INFO[expected_cause]["reason_phrase"] in appeal_letter
    else:
        cited = any(info["reason_phrase"] in appeal_letter for info in CAUSE_INFO.values())
    return {
        "cites_denial_reason": cited,
        "invented_codes": sorted(invented_codes),
        "invented_policy_numbers": sorted(invented_policies),
        "has_reconsideration_request": "reconsider" in appeal_letter.lower()
        or "reprocess" in appeal_letter.lower(),
        "passes": cited and not invented_codes and not invented_policies,
    }
