from denial_appeal.evaluate import (appeal_quality_metrics,
                                     calibration_metrics,
                                     classification_metrics)
from denial_appeal.quality import check_appeal, extract_codes
from denial_appeal.rules import classify_rules

def test_rules_classify_each_category(records):
    by_cause = {}
    for r in records:
        by_cause.setdefault(r["cause"], r)
    for cause, r in by_cause.items():
        assert classify_rules(r["denial_letter"]) == cause

def test_quality_ground_truth_passes(records):
    r = records[0]
    assert check_appeal(r["denial_letter"], r["appeal_letter"], r["cause"])["passes"]

def test_quality_flags_invented_code(records):
    r = records[0]
    bad = r["appeal_letter"] + "\nSee also denial code DEN-999."
    assert check_appeal(r["denial_letter"], bad, r["cause"])["invented_codes"] == ["DEN-999"]

def test_quality_flags_invented_policy(records):
    r = records[0]
    bad = r["appeal_letter"] + "\nPer Policy 99.88.77 payment is required."
    result = check_appeal(r["denial_letter"], bad, r["cause"])
    assert result["invented_policy_numbers"] and not result["passes"]

def test_extract_codes():
    assert extract_codes("codes DEN-50 and DEN-97, not DENX") == {"DEN-50", "DEN-97"}

def test_classification_metrics_perfect():
    y = ["eligibility", "timely_filing"]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["n"] == 2

def test_classification_metrics_wrong():
    m = classification_metrics(["eligibility"], ["timely_filing"])
    assert m["accuracy"] == 0.0

def test_calibration_structure(records):
    cal = calibration_metrics(records)
    assert 0.0 <= cal["brier_score"] <= 1.0
    assert set(cal["by_band"]) == {"low", "medium", "high"}
    assert "simulated" in cal["note"]

def test_appeal_quality_metrics_ground_truth(records):
    q = appeal_quality_metrics(records)
    assert q["pass_rate"] == 1.0 and q["invented_code_rate"] == 0.0
