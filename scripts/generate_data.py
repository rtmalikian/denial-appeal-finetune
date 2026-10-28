#!/usr/bin/env python3
"""Generate the committed synthetic sample + SFT / preference exports."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from denial_appeal.formats import to_preferences, to_sft
from denial_appeal.generator import generate_records, write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=600)
    ap.add_argument("--test-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "sample"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train = generate_records(a.train_n, seed=a.seed)
    test = generate_records(a.test_n, seed=a.seed + 1000)
    hashes = {}
    hashes["train.jsonl"] = write_jsonl(train, str(out / "train.jsonl"))
    hashes["test.jsonl"] = write_jsonl(test, str(out / "test.jsonl"))
    hashes["sft_train.jsonl"] = write_jsonl(to_sft(train), str(out / "sft_train.jsonl"))
    hashes["preferences_train.jsonl"] = write_jsonl(to_preferences(train), str(out / "preferences_train.jsonl"))
    (out / "hashes.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps({"train": len(train), "test": len(test), "hashes": hashes}, indent=2))

if __name__ == "__main__":
    main()
