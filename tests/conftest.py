import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from denial_appeal.generator import generate_records

@pytest.fixture(scope="session")
def records():
    return generate_records(80, seed=7)

@pytest.fixture(scope="session")
def tiny_train(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    from denial_appeal.generator import write_jsonl
    write_jsonl(generate_records(80, seed=11), str(d / "train.jsonl"))
    write_jsonl(generate_records(40, seed=1011), str(d / "test.jsonl"))
    return d
