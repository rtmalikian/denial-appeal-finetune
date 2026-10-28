from collections import Counter
from denial_appeal.categories import CATEGORIES
from denial_appeal.generator import generate_records

def test_deterministic_same_seed():
    assert generate_records(20, seed=3) == generate_records(20, seed=3)

def test_different_seed_differs():
    assert generate_records(20, seed=3) != generate_records(20, seed=4)

def test_all_categories_present():
    recs = generate_records(80, seed=1)
    assert set(r["cause"] for r in recs) == set(CATEGORIES)

def test_at_least_eight_categories():
    assert len(CATEGORIES) >= 8

def test_balanced_categories():
    counts = Counter(r["cause"] for r in generate_records(80, seed=1))
    assert max(counts.values()) - min(counts.values()) <= 1

def test_records_labelled_synthetic(records):
    assert all(r["synthetic"] is True for r in records)
    assert all("synthetic" in r["denial_letter"].lower() for r in records)

def test_schema_fields(records):
    required = {"id", "denial_letter", "cause", "appeal_success_probability",
                "appeal_success_probability_band", "appeal_letter"}
    assert all(required <= set(r) for r in records)

def test_probability_band_consistency(records):
    from denial_appeal.categories import band_for
    assert all(band_for(r["appeal_success_probability"]) ==
               r["appeal_success_probability_band"] for r in records)

def test_probabilities_in_range(records):
    assert all(0.0 < r["appeal_success_probability"] < 1.0 for r in records)

def test_unique_ids(records):
    assert len({r["id"] for r in records}) == len(records)

def test_invalid_n():
    import pytest
    with pytest.raises(ValueError):
        generate_records(0)
