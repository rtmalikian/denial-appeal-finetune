# Results — what was actually run (2026-10-28)

All numbers below were measured in this workspace on **synthetic** data.
Nothing here is a real-world result. Raw output: `artifacts/metrics.json`,
`artifacts/benchmark.json`, `artifacts/run_manifest.json`.

## 1. Denial-cause classification — synthetic held-out test (n=150, train n=600, seed 42)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Keyword-rule baseline | 1.000 | 1.000 |
| Trained TF-IDF + Logistic Regression | 1.000 | 1.000 |
| Code-masked test (DEN-### → DEN-XXX): rules | 1.000 | 1.000 |
| Code-masked test: trained model | 1.000 | 1.000 |

Per-cause precision/recall/F1 are 1.000 for every cause for both systems
(full table in `artifacts/metrics.json`).

**Read this honestly:** the scores saturate because templated synthetic
letters are separable by construction — each cause has distinctive language,
and that stays true even with the denial codes masked. These numbers prove
the data → training → evaluation pipeline is correct end-to-end. They are
**not** evidence about performance on real denial letters, where language
overlaps, causes compound, and payer wording varies. The useful comparison
this repo *does* establish is methodological: a trained model, a rules
baseline, a masked ablation, per-cause metrics, and a regression gate —
the same harness a real dataset would plug into.

## 2. Probability-band calibration (simulated outcomes, test n=150)

| Band | n | Mean stated probability | Empirical (simulated) success rate |
|---|---|---|---|
| low | 43 | 0.284 | 0.209 |
| medium | 67 | 0.536 | 0.508 |
| high | 40 | 0.744 | 0.725 |

Brier score 0.2153; band-level ECE 0.0391. Outcomes are simulated by the
generator, so this checks the generator's internal consistency only.

## 3. Appeal-draft quality (ground-truth appeals, test n=150)

Pass rate 1.000; denial-reason citation rate 1.000; invented-code rate
0.000; invented-policy-number rate 0.000 (checker: `src/denial_appeal/quality.py`).
The preference-pair rejected examples fail this checker by construction
(invented code DEN-999, wrong cited reason) — verified in tests.

## 4. Serving benchmark — CPU fallback server (measured, local loopback)

Against `denial_appeal.serve_stub` (TF-IDF classifier + template appeal,
NOT an LLM), 50 requests after 5 warmup, via `bench/benchmark.py`:

| Metric | Value |
|---|---|
| p50 latency | 2.93 ms |
| p95 latency | 7.72 ms |
| Mean latency | 3.71 ms |
| Throughput | 265.0 req/s |

These are classifier/template numbers on loopback. They say nothing about
vLLM/LLM generation latency, which must be measured on a GPU with
`serve_vllm.py` + the same benchmark script.

## 5. LoRA SFT — smoke status: BLOCKED in this sandbox (honest record)

- `train_lora.py --dry-run`: **passed** — config and 600-example SFT data
  validated; plan printed (Qwen2.5-0.5B-Instruct, LoRA r=16/α=32, lr 2e-4).
- `train_lora.py --smoke`: **blocked, no steps run.** Attempts:
  (1) `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  failed — the PyTorch index connection was reset (`RemoteDisconnected`)
  and no distribution was fetched; (2) `pip install transformers peft
  datasets` from PyPI failed with `OSError: [Errno 28] No space left on
  device` on this sandbox's small `/tmp` tmpfs during download.
  The script therefore reported `{"smoke_status": "blocked", "reason":
  "GPU training dependencies unavailable: No module named 'torch'."}` and
  exited 0 in smoke mode, by design.
- **No LoRA metrics are claimed anywhere in this repo.** On a GPU machine:
  `pip install -r requirements-gpu.txt && make lora-smoke`.

## 6. Tests

`pytest`: **35 passed** (see README). Regression gate
(`scripts/evaluate_cli.py`): passes — trained accuracy ≥ rules accuracy and
ground-truth appeal-quality pass rate ≥ 0.99.
