#!/usr/bin/env python3
"""Regression gate: fail (exit 1) if trained model does not beat rules or
ground-truth appeal quality drops. Used by CI and the Makefile."""
import json, sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts/metrics.json").read_text())
    trained, rules = metrics["trained_model"], metrics["rules_baseline"]
    quality = metrics["appeal_quality_test_ground_truth"]
    ok = trained["accuracy"] >= rules["accuracy"] and quality["pass_rate"] >= 0.99
    print(json.dumps({"trained_accuracy": trained["accuracy"],
                      "rules_accuracy": rules["accuracy"],
                      "appeal_quality_pass_rate": quality["pass_rate"],
                      "gate_passes": ok}, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
