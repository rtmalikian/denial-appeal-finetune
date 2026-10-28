"""Evaluation harness: per-cause P/R/F1, calibration, appeal quality."""
from __future__ import annotations

from collections import defaultdict

from .categories import CATEGORIES, BANDS
from .quality import check_appeal


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(CATEGORIES)
    per = {}
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    present = set(y_true) | set(y_pred)
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if lab in present:  # macro-F1 over labels present in truth or prediction
            f1s.append(f1)
        per[lab] = {"precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1, 4), "support": tp + fn}
    return {
        "accuracy": round(correct / len(y_true), 4) if y_true else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "n": len(y_true),
        "per_cause": per,
    }


def calibration_metrics(records: list[dict]) -> dict:
    """Compare stated probability vs simulated outcome, by band and overall.
    Outcomes are simulated by construction — this checks internal calibration
    of the synthetic generator, NOT real-world appeal calibration."""
    by_band = {}
    sq_err, abs_err = 0.0, 0.0
    for band in BANDS:
        rows = [r for r in records if r["appeal_success_probability_band"] == band]
        if not rows:
            by_band[band] = {"n": 0, "mean_probability": None, "empirical_rate": None}
            continue
        mp = sum(r["appeal_success_probability"] for r in rows) / len(rows)
        er = sum(1 for r in rows if r["simulated_appeal_succeeded"]) / len(rows)
        by_band[band] = {"n": len(rows), "mean_probability": round(mp, 4),
                         "empirical_rate": round(er, 4)}
    for r in records:
        err = r["appeal_success_probability"] - (1.0 if r["simulated_appeal_succeeded"] else 0.0)
        sq_err += err * err; abs_err += abs(r["appeal_success_probability"] -
                                             (1.0 if r["simulated_appeal_succeeded"] else 0.0))
    n = len(records) or 1
    # ECE across the 3 bands
    ece = 0.0
    for band in BANDS:
        b = by_band[band]
        if b["n"]:
            ece += (b["n"] / n) * abs(b["mean_probability"] - b["empirical_rate"])
    return {"by_band": by_band, "brier_score": round(sq_err / n, 4),
            "expected_calibration_error_bands": round(ece, 4), "n": len(records),
            "note": "Outcomes are simulated; calibration is internal to the synthetic generator."}


def appeal_quality_metrics(records: list[dict]) -> dict:
    checks = [check_appeal(r["denial_letter"], r["appeal_letter"], r["cause"]) for r in records]
    n = len(checks) or 1
    return {
        "n": len(checks),
        "pass_rate": round(sum(c["passes"] for c in checks) / n, 4),
        "cite_rate": round(sum(c["cites_denial_reason"] for c in checks) / n, 4),
        "invented_code_rate": round(sum(bool(c["invented_codes"]) for c in checks) / n, 4),
        "invented_policy_rate": round(sum(bool(c["invented_policy_numbers"]) for c in checks) / n, 4),
    }
