"""Deterministic SYNTHETIC denial/appeal dataset generator.

Every record is synthetic. No real insurer, member, claim, or policy data is
used or imitated. The generator reproduces, at small open scale, the
methodology described in DATASET_CARD.md: templated denial letters across a
fixed cause taxonomy, structured labels assigned by construction (the cause
that generated the letter IS the label), a template appeal that cites the
denial reason verbatim, and a simulated appeal outcome drawn from the
record's success probability so probability bands can be calibrated.

Human-in-the-loop note: in the original 10k workflow a human reviewed
samples and corrected labelling rules; here the rules are the code below and
are fully inspectable.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable

from .categories import CATEGORIES, CAUSE_INFO, band_for

SERVICES = [
    "outpatient physical therapy visit", "MRI lumbar spine without contrast",
    "specialty office visit", "laboratory panel", "outpatient surgical procedure",
    "behavioural health session", "durable medical equipment rental",
    "infusion administration", "diagnostic colonoscopy", "cardiac stress test",
]

_TEMPLATES = [
    ("Dear Provider,\n\nClaim {claim} for {service} on {date} for member "
     "{member} has been denied (code {code}). Reason: the service is "
     "{reason_phrase}. {extra}\n\nYou may request reconsideration in writing. "
     "This is a synthetic denial letter generated for software demonstration."),
    ("RE: Claim {claim} — Denial Notice\n\nService: {service}. Date of service: "
     "{date}. Member: {member}.\n\nDetermination: Denied, code {code}. "
     "This claim is denied because it is {reason_phrase}. {extra}\n\n"
     "Appeals must include the claim number. SYNTHETIC DATA — not a real denial."),
    ("Explanation of Benefits — Claim {claim}\n\n{service} ({date}), member "
     "{member}: DENIED ({code}). {reason_sentence} {extra}\n\n"
     "If you disagree, submit a written appeal. This letter is synthetic."),
]

_EXTRAS = {
    "medical_necessity": [
        "The submitted notes did not show that clinical criteria were met after conservative treatment.",
        "Review found the medical necessity documentation did not support the level of service billed.",
    ],
    "prior_auth_missing": [
        "No authorization on file was found for this date of service; auth was not obtained before the visit.",
        "Our records show prior authorization was required and none was requested.",
    ],
    "coding_bundling": [
        "The procedure is bundled into the primary service under a bundling edit; a modifier would be required for a distinct service.",
        "This line is mutually exclusive with the primary procedure billed the same day.",
    ],
    "eligibility": [
        "Coverage was not in effect on the date of service; the member was not covered under this plan.",
        "Eligibility records show coverage terminated before this visit.",
    ],
    "timely_filing": [
        "The claim was filed late, after the submission window closed.",
        "Proof of timely filing was not provided with the claim.",
    ],
    "duplicate_claim": [
        "A claim for the same service was already paid; this appears to be a previously submitted line.",
        "The same service was already paid under a prior claim number.",
    ],
    "documentation_insufficient": [
        "Additional information is required: records were not received with the claim.",
        "The submission lacks information needed to adjudicate this line.",
    ],
    "experimental_investigational": [
        "The procedure is not proven effective for this indication under the plan's investigational definition.",
        "Available evidence was reviewed and the service remains investigational.",
    ],
}


def _appeal_letter(rec_cause: str, claim: str, service: str, code: str) -> str:
    info = CAUSE_INFO[rec_cause]
    return (
        f"Dear Appeals Department,\n\n"
        f"I am writing to appeal the denial of claim {claim} for {service} "
        f"(denial code {code}). The denial states the service is "
        f"{info['reason_phrase']}.\n\n"
        f"Enclosed is {info['evidence_to_attach']}. Please reconsider the "
        f"denial in light of this documentation and reprocess the claim.\n\n"
        f"If any further information is required, please contact our office. "
        f"This appeal was generated from synthetic data for demonstration; a "
        f"human reviewer must verify it before any real use.\n\n"
        f"Sincerely,\nBilling Office (synthetic)"
    )


def generate_records(n: int, seed: int = 42) -> list[dict]:
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    causes = [CATEGORIES[i % len(CATEGORIES)] for i in range(n)]
    rng.shuffle(causes)
    records = []
    for i, cause in enumerate(causes):
        info = CAUSE_INFO[cause]
        local = random.Random(rng.random())
        service = local.choice(SERVICES)
        claim = f"SYN-{seed:04d}-{i:05d}"
        member = f"SYNTHETIC-{local.randint(10000, 99999)}"
        date = f"2026-{local.randint(1, 9):02d}-{local.randint(1, 28):02d}"
        extra = local.choice(_EXTRAS[cause])
        tpl = _TEMPLATES[i % len(_TEMPLATES)]
        reason_sentence = (
            f"The reason given was: {info['reason_phrase']}."
        )
        letter = tpl.format(
            claim=claim, service=service, date=date, member=member,
            code=info["denial_code"], reason_phrase=info["reason_phrase"],
            extra=extra, reason_sentence=reason_sentence,
        )
        # Success probability: cause base +/- noise, clipped.
        prob = min(0.95, max(0.05, info["base_success"] + local.uniform(-0.18, 0.18)))
        prob = round(prob, 3)
        outcome = local.random() < prob  # simulated outcome, for calibration only
        records.append({
            "id": f"syn-{seed}-{i:05d}",
            "synthetic": True,
            "denial_letter": letter,
            "cause": cause,
            "cause_label": info["label"],
            "denial_code": info["denial_code"],
            "service": service,
            "appeal_success_probability": prob,
            "appeal_success_probability_band": band_for(prob),
            "simulated_appeal_succeeded": bool(outcome),
            "appeal_letter": _appeal_letter(cause, claim, service, info["denial_code"]),
        })
    return records


def write_jsonl(records: Iterable[dict], path: str) -> str:
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=False, sort_keys=True)
            h.update(line.encode()); h.update(b"\n")
            f.write(line + "\n")
    return h.hexdigest()


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
