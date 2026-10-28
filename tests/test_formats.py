import json
from denial_appeal.formats import (instruction_example, preference_example,
                                   sft_example, to_preferences, to_sft)
from denial_appeal.quality import check_appeal

def test_sft_roles(records):
    ex = sft_example(records[0])
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]

def test_sft_assistant_is_valid_json(records):
    payload = json.loads(sft_example(records[0])["messages"][2]["content"])
    assert payload["cause"] == records[0]["cause"]
    assert payload["appeal_letter"] == records[0]["appeal_letter"]

def test_sft_user_contains_letter(records):
    ex = sft_example(records[0])
    assert records[0]["denial_letter"] in ex["messages"][1]["content"]

def test_instruction_format(records):
    ex = instruction_example(records[0])
    assert set(ex) == {"instruction", "input", "output"}

def test_preference_chosen_passes_quality(records):
    ex = preference_example(records[0])
    assert check_appeal(records[0]["denial_letter"], ex["chosen"], records[0]["cause"])["passes"]

def test_preference_rejected_fails_quality(records):
    ex = preference_example(records[0])
    result = check_appeal(records[0]["denial_letter"], ex["rejected"], records[0]["cause"])
    assert not result["passes"]
    assert result["invented_codes"]  # DEN-999 is invented

def test_preference_chosen_differs_from_rejected(records):
    ex = preference_example(records[0])
    assert ex["chosen"] != ex["rejected"]

def test_bulk_counts(records):
    assert len(to_sft(records)) == len(records)
    assert len(to_preferences(records)) == len(records)
