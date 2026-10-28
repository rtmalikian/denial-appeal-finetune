# Model Card — Denial Appeal Fine-Tune (proof of concept)

## Models in this repo

1. **CPU baseline (actually trained here)** — TF-IDF (1–2 grams) +
   Logistic Regression for denial-cause classification, trained on the
   committed 600-record synthetic training split and evaluated on the
   150-record synthetic held-out split. Artifact:
   `artifacts/baseline_model.pkl`; metrics: `artifacts/metrics.json` and
   `RESULTS.md`.
2. **LoRA SFT pipeline (code provided, not trained here)** —
   `train_lora.py` + `configs/lora_qwen25_*.yaml` fine-tune
   Qwen2.5-0.5B/1.5B-Instruct with PEFT LoRA on the chat-format SFT export
   for the joint task (classify + band + draft appeal). See RESULTS.md for
   the recorded smoke-test status in this environment. No LoRA metrics are
   claimed anywhere in this repo.

## Intended use

- Revenue-cycle **workflow triage demonstration**: route a denial letter to
  a cause category, surface a success-probability band as a prioritisation
  signal, and produce a first-draft appeal for a human biller to edit.
- Education and portfolio demonstration of an end-to-end fine-tuning data /
  training / serving / evaluation workflow on synthetic data.

## Out of scope / not for

- **Not for submitting real appeals without human review.** Every draft
  must be verified by a qualified billing professional against the real
  denial, the real contract, and real documentation before use.
- No clinical or medication recommendations — this is revenue-cycle
  workflow software, not clinical decision support.
- Not validated on real denial letters, real payers, or real outcomes.
  Synthetic held-out accuracy does not transfer by assumption.
- Success-probability bands are synthetic priors, not predictions about any
  real claim.

## Risks and mitigations built in

- **Invented codes / policy numbers in drafts** → `quality.py` checker
  flags any denial code or policy number in an appeal that is absent from
  the source letter; the eval harness reports invention rates and CI gates
  on the ground-truth pass rate.
- **Automation bias** → outputs carry a human-review note; the server
  labels responses synthetic.
- **Privacy** → the pipeline is designed for synthetic data here; a real
  deployment would require a HIPAA-appropriate environment, a BAA-covered
  model endpoint, de-identification, access controls, and audit logging —
  none of which this proof of concept provides.

## Author

Raphael Malikian <rtmalikian@gmail.com> — healthcare founder/operator
(bootstrapped and ran a direct primary care clinic as CEO), now building
clinical and revenue-cycle AI.
