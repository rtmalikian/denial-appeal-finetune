"""SFT and preference-pair (DPO-style) formatting.

SFT: OpenAI-style chat messages JSONL, plus an instruction/response variant.
Preference: prompt / chosen / rejected triples where `chosen` is the
ground-truth appeal and `rejected` is a deliberately degraded appeal
(wrong cited reason and/or an invented denial code). Rejected examples are
synthetic negatives constructed for training-data demonstration — they are
never presented as model output or real appeals.
"""
from __future__ import annotations

import json

from .categories import CATEGORIES, CAUSE_INFO

SYSTEM = (
    "You analyse synthetic insurance denial letters for a revenue-cycle "
    "workflow. Identify the denial cause, estimate the appeal success "
    "probability band, and draft an appeal that cites the denial reason "
    "verbatim and invents no codes or policy numbers. A human reviews every "
    "draft before use."
)
TASK = (
    "Analyse this denial letter. Respond with JSON containing cause, "
    "appeal_success_probability_band, and appeal_letter.\n\n{letter}"
)


def sft_example(record: dict) -> dict:
    assistant = json.dumps({
        "cause": record["cause"],
        "appeal_success_probability_band": record["appeal_success_probability_band"],
        "appeal_letter": record["appeal_letter"],
    }, ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK.format(letter=record["denial_letter"])},
        {"role": "assistant", "content": assistant},
    ]}


def instruction_example(record: dict) -> dict:
    ex = sft_example(record)
    return {
        "instruction": ex["messages"][1]["content"],
        "input": "",
        "output": ex["messages"][2]["content"],
    }


def _degraded_appeal(record: dict) -> str:
    # Pick a different cause and an invented code — a clear negative.
    other = next(c for c in CATEGORIES if c != record["cause"])
    return (
        "Dear Appeals Department,\n\n"
        f"Please reconsider claim under denial code DEN-999. The denial "
        f"states the service is {CAUSE_INFO[other]['reason_phrase']}. "
        f"Per Policy 12.34.56 this must be paid. Please reprocess.\n\n"
        "Sincerely,\nBilling Office (synthetic)"
    )


def preference_example(record: dict) -> dict:
    return {
        "prompt": TASK.format(letter=record["denial_letter"]),
        "chosen": record["appeal_letter"],
        "rejected": _degraded_appeal(record),
        "synthetic": True,
    }


def to_sft(records): return [sft_example(r) for r in records]
def to_instructions(records): return [instruction_example(r) for r in records]
def to_preferences(records): return [preference_example(r) for r in records]
