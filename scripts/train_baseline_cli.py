#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from denial_appeal.train_baseline import train_and_evaluate

def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(root / "data/sample/train.jsonl"))
    ap.add_argument("--test", default=str(root / "data/sample/test.jsonl"))
    ap.add_argument("--out", default=str(root / "artifacts"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(train_and_evaluate(a.train, a.test, a.out, a.seed), indent=2))

if __name__ == "__main__":
    main()
