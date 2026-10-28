> ## ⚠️ PROOF OF CONCEPT — NOT CLINICALLY TESTED
>
> **This tool has not been tested clinically and is a proof-of-concept for
> research and software demonstration only. It is not a medical device, it
> does not diagnose, treat, or rule out any condition, and its outputs may
> be wrong — including false positives and false negatives.**
>
> **Do not use it to make decisions about your health.**
> For any medical issue or concern, seek a licensed healthcare professional.
> **In an emergency, go to your nearest emergency room or dial 911 immediately.**

> Additional limits for this repo: this is **revenue-cycle workflow
> software**, not clinical software. All data is **synthetic**. Appeal
> drafts must be verified by a qualified billing professional before any
> real use — see `MODEL_CARD.md`.

# Denial Appeal Fine-Tune — LoRA SFT for Insurance Denial Classification & Appeal Drafting | Clinical RCM AI, PyTorch, vLLM

**Denial appeal automation with fine-tuning, done in the open:** a synthetic
denial/appeal dataset generator, SFT and DPO-style preference data, a real
CPU-trained classification baseline, a full PyTorch + PEFT **LoRA
fine-tuning** pipeline for Qwen2.5-Instruct, an OpenAI-compatible **vLLM**
serving entrypoint (plus a CPU fallback server with the same API), and an
evaluation harness with per-cause precision/recall, probability-band
calibration, and a hallucinated-code checker for appeal drafts.

Keywords: insurance denial appeal automation · revenue cycle management
(RCM) AI · medical billing AI · denial classification · appeal letter
generation · LoRA fine-tuning · supervised fine-tuning (SFT) · DPO
preference data · PyTorch · Hugging Face Transformers · PEFT · vLLM
serving · LLM evaluation · healthcare AI

## The measurable workflow problem

A biller triaging a denial by hand has to read the letter, work out *why*
it was denied, decide whether an appeal is worth the time, and draft the
appeal — commonly tens of minutes per letter before any documentation is
even gathered. This repo automates the first three steps and drafts the
fourth:

denial letter → **denial-cause classification** (8 causes) →
**appeal-success probability band** (triage/prioritisation) → **appeal
draft that cites the denial reason verbatim and invents no codes** →
human review.

The fallback server answers that pipeline in **p50 2.93 ms / p95 7.72 ms**
locally (classifier + template; measured, see `RESULTS.md`) — the point is
not the milliseconds, it's that triage becomes a sorting problem instead of
a reading problem.

## Background

This reproduces, as open runnable code, a workflow Raphael built for
medical billing: a Qwen model workflow that analysed denial letters,
identified the denial cause, drafted appeals, and estimated appeal
success, trained on **10,000 synthetic denial/appeal letters generated
with Codex under human-in-the-loop review**. The original generation
methodology, schema, and labelling rules are documented in
`DATASET_CARD.md` and re-implemented in
`src/denial_appeal/generator.py` at a committed sample scale
(600 train / 150 held-out test) that anyone can regenerate, hash, and
scale back to 10k with one command.

## Quick start (CPU, no GPU, no downloads beyond pip)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data        # regenerate the synthetic sample + SFT/preference exports
make train       # train the TF-IDF + Logistic Regression baseline (real run)
make eval        # regression gate vs the keyword-rule baseline
make test        # 35 tests
make serve       # OpenAI-compatible fallback server on :8000
make bench       # latency/throughput benchmark against the running server
```

## GPU: LoRA fine-tuning + vLLM serving

```bash
pip install -r requirements-gpu.txt
make lora-dry-run   # validate config + data, print the training plan
make lora-smoke     # 3 LoRA steps on 8 examples (Qwen2.5-0.5B-Instruct)
python train_lora.py --config configs/lora_qwen25_15b.yaml   # full run
python serve_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter artifacts/lora-qwen25-05b
```

Configs for Qwen2.5-0.5B and 1.5B-Instruct live in `configs/` (LoRA
r=16/α=32, gradient checkpointing, seed-controlled, Trainer eval loop).

## Results (synthetic held-out, measured — full detail in RESULTS.md)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Keyword-rule baseline | 1.000 | 1.000 |
| Trained TF-IDF + Logistic Regression | 1.000 | 1.000 |

Appeal-draft quality on ground-truth appeals: pass rate 1.000, invented
codes 0, invented policy numbers 0. Calibration (simulated outcomes):
Brier 0.2153, band ECE 0.0391.

## What this proves / what it does not

**Proves:** an end-to-end, reproducible fine-tuning workflow — synthetic
data generation with labelling-by-construction, SFT + preference data
formatting, a trained baseline with a run manifest (data hashes, seed,
hyperparameters), per-cause and calibration evals, a hallucination checker
for generated drafts, and dual serving (vLLM / CPU fallback) behind one
OpenAI-compatible interface, benchmarked with the same script.

**Does not prove:** performance on real denial letters. Templated
synthetic language is separable by construction, so the classification
scores above saturate — they validate the pipeline, not real-world
accuracy. **No LoRA training run is claimed in this repo:** in the
sandbox where it was built, the LoRA smoke test was blocked (PyTorch
download failed; see RESULTS.md §5 for the exact record). No real insurer
data, real claims, or real appeal outcomes appear anywhere in this repo.

## Layout

```
src/denial_appeal/   generator, taxonomy, SFT/preference formats, rules
                     baseline, training, evals, quality checker, server
data/sample/         600 train / 150 test JSONL + SFT + preference exports
configs/             baseline + LoRA configs (0.5B / 1.5B)
train_lora.py        PyTorch/Transformers/PEFT LoRA SFT (+ --dry-run/--smoke)
serve_vllm.py        vLLM OpenAI-compatible serving entrypoint
bench/benchmark.py   latency/throughput benchmark for either server
artifacts/           trained baseline, metrics.json, run_manifest.json, benchmark.json
tests/               35 pytest tests
DATASET_CARD.md · MODEL_CARD.md · RESULTS.md
```

## Author

**Raphael Malikian** <rtmalikian@gmail.com> — healthcare founder/operator
(bootstrapped and ran a direct primary care clinic as CEO), now building
clinical and revenue-cycle AI. Open to collaboration on denial-management,
RCM AI, and clinical LLM evaluation work.

*AI assistance was used in building this repository; all results were
actually run and are recorded in RESULTS.md.*
