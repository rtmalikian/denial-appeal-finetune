"""Train the CPU baseline: TF-IDF + Logistic Regression cause classifier.

This is a REAL training run on the synthetic data: it fits, evaluates rules
vs trained model on the held-out test split, writes the model artifact, a
metrics file, and a run manifest (data hashes, hyperparams, seed).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from .evaluate import appeal_quality_metrics, calibration_metrics, classification_metrics
from .generator import read_jsonl
from .rules import classify_rules


def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0, random_state=0)),
    ])


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_and_evaluate(train_path: str, test_path: str, out_dir: str,
                       seed: int = 42) -> dict:
    train, test = read_jsonl(train_path), read_jsonl(test_path)
    model = build_pipeline()
    model.fit([r["denial_letter"] for r in train], [r["cause"] for r in train])
    y_true = [r["cause"] for r in test]
    y_pred = list(model.predict([r["denial_letter"] for r in test]))
    y_rules = [classify_rules(r["denial_letter"]) for r in test]
    import re
    mask = lambda t: re.sub(r"DEN-\d+", "DEN-XXX", t)  # noqa: E731
    results = {
        "synthetic_data": True,
        "train_n": len(train), "test_n": len(test), "seed": seed,
        "rules_baseline": classification_metrics(y_true, y_rules),
        "trained_model": classification_metrics(y_true, y_pred),
        "code_masked_evaluation": {
            "note": "Denial codes replaced with DEN-XXX at test time. Even "
                    "masked, templated synthetic language is separable by "
                    "construction — saturation here proves pipeline "
                    "correctness, not real-world performance.",
            "rules_baseline": classification_metrics(
                y_true, [classify_rules(mask(r["denial_letter"])) for r in test]),
            "trained_model_original": classification_metrics(
                y_true, list(model.predict([mask(r["denial_letter"]) for r in test]))),
        },
        "calibration_test": calibration_metrics(test),
        "appeal_quality_test_ground_truth": appeal_quality_metrics(test),
    }
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "baseline_model.pkl", "wb") as f:
        pickle.dump(model, f)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    manifest = {
        "run_type": "tfidf_logreg_baseline_cpu",
        "git_sha": "LOCAL_UNPUSHED",
        "seed": seed,
        "train_data_sha256": _sha(train_path), "test_data_sha256": _sha(test_path),
        "hyperparams": {"vectorizer": "tfidf(1,2) sublinear", "classifier": "LogisticRegression(C=4.0,max_iter=1000)"},
        "metrics_file": "metrics.json",
        "synthetic_data": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    results["manifest"] = manifest
    return results


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
