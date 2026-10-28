"""Denial cause taxonomy. All categories / codes here are SYNTHETIC workflow
labels for this proof-of-concept. They are not a real payer's code set."""
from __future__ import annotations

CATEGORIES = [
    "medical_necessity",
    "prior_auth_missing",
    "coding_bundling",
    "eligibility",
    "timely_filing",
    "duplicate_claim",
    "documentation_insufficient",
    "experimental_investigational",
]

CAUSE_INFO = {
    "medical_necessity": {
        "label": "Medical necessity not established",
        "denial_code": "DEN-50",
        "reason_phrase": "not deemed a medical necessity",
        "evidence_to_attach": "treating clinician notes and peer-reviewed guideline excerpts already in the chart",
        "base_success": 0.45,
        "keywords": ["medical necessity", "not deemed a medical necessity", "clinical criteria", "conservative treatment"],
    },
    "prior_auth_missing": {
        "label": "Prior authorization missing",
        "denial_code": "DEN-197",
        "reason_phrase": "precertification absent",
        "evidence_to_attach": "the authorization request log and proof of timely submission",
        "base_success": 0.70,
        "keywords": ["precertification absent", "prior authorization", "no authorization on file", "auth was not obtained"],
    },
    "coding_bundling": {
        "label": "Coding / bundling edit",
        "denial_code": "DEN-97",
        "reason_phrase": "benefit for this service is included in the payment for another service",
        "evidence_to_attach": "the operative note showing a distinct and separate service",
        "base_success": 0.60,
        "keywords": ["included in the payment", "bundled", "bundling", "mutually exclusive", "modifier"],
    },
    "eligibility": {
        "label": "Member not eligible on date of service",
        "denial_code": "DEN-27",
        "reason_phrase": "expenses incurred after coverage terminated",
        "evidence_to_attach": "eligibility verification records from the date of service",
        "base_success": 0.35,
        "keywords": ["coverage terminated", "not eligible", "coverage was not in effect", "member was not covered"],
    },
    "timely_filing": {
        "label": "Timely filing limit exceeded",
        "denial_code": "DEN-29",
        "reason_phrase": "the time limit for filing has expired",
        "evidence_to_attach": "the clearinghouse acceptance report showing the original submission date",
        "base_success": 0.55,
        "keywords": ["time limit for filing has expired", "timely filing", "filed late", "submission window"],
    },
    "duplicate_claim": {
        "label": "Duplicate claim",
        "denial_code": "DEN-18",
        "reason_phrase": "exact duplicate claim or service",
        "evidence_to_attach": "a line-by-line comparison showing this is a corrected, not duplicate, submission",
        "base_success": 0.50,
        "keywords": ["exact duplicate", "duplicate claim", "previously submitted", "same service was already paid"],
    },
    "documentation_insufficient": {
        "label": "Documentation insufficient",
        "denial_code": "DEN-16",
        "reason_phrase": "claim lacks information or has submission errors",
        "evidence_to_attach": "the complete clinical documentation requested in the denial notice",
        "base_success": 0.75,
        "keywords": ["lacks information", "insufficient documentation", "records were not received", "additional information is required"],
    },
    "experimental_investigational": {
        "label": "Experimental / investigational",
        "denial_code": "DEN-55",
        "reason_phrase": "procedure is considered experimental or investigational",
        "evidence_to_attach": "published evidence summaries already referenced in the original request",
        "base_success": 0.25,
        "keywords": ["experimental or investigational", "investigational", "not proven effective", "experimental procedure"],
    },
}

BANDS = ("low", "medium", "high")

def band_for(probability: float) -> str:
    if probability < 0.40:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"
